const { Container, Graphics, Rectangle, RenderTexture, RendererType, UniformGroup } = require('pixi.js');
const { KawaseBlurFilter } = require('../../lib');
const assert = require('node:assert/strict');

module.exports = async function verifyKawaseBlur(renderer)
{
    // Observe the values submitted to the filter manager even when CI has no WebGPU adapter.
    for (const quality of [1, 2, 3, 4])
    {
        const filter = new KawaseBlurFilter({ strength: 8, quality });
        const group = filter.resources.kawaseBlurUniforms;
        const input = RenderTexture.create({ width: 32, height: 16 });
        const snapshots = [];
        let submitted = 0;
        const manager = {
            renderer: {
                type: RendererType.WEBGPU,
                renderPipes: {
                    uniformBatch: {
                        getUboResource(uniformGroup)
                        {
                            const snapshot = new UniformGroup({
                                uOffset: { value: uniformGroup.uniforms.uOffset.slice(), type: 'vec2<f32>' },
                            });

                            snapshots.push(snapshot);

                            return snapshot;
                        },
                    },
                },
            },
            applyFilter(applied)
            {
                assert.equal(snapshots.length, submitted + 1);
                assert.equal(applied.groups[1].resources[0], snapshots[submitted++]);
            },
        };

        filter.apply(manager, input, input, true);
        assert.equal(submitted, quality);
        snapshots.forEach((snapshot, index) =>
        {
            const offset = filter.kernels[index] + 0.5;

            assert.deepEqual(Array.from(snapshot.uniforms.uOffset), [Math.fround(offset / 32), Math.fround(offset / 16)]);
        });
        assert.equal(filter.resources.kawaseBlurUniforms, group);
        assert.equal(filter.uniforms, group.uniforms);

        // A failed draw must not leave a renderer-specific resource in the public slot.
        filter.quality = 1;
        manager.applyFilter = () => { throw new Error('draw failed'); };
        assert.throws(() => filter.apply(manager, input, input, true), /draw failed/);
        assert.equal(filter.resources.kawaseBlurUniforms, group);
        filter.destroy();
        input.destroy(true);
    }

    const width = 96;
    const height = 96;
    const canvas = document.createElement('canvas');

    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });

    async function render(filters)
    {
        const scene = new Container();
        const content = new Container();

        scene.addChild(new Graphics().rect(0, 0, width, height).fill({ color: 0x000000 }), content);
        content.addChild(new Graphics().rect(40, 36, 16, 24).fill(0xffffff));
        content.filterArea = new Rectangle(0, 0, width, height);
        content.filters = filters;
        // Keep the frame identical for a multipass filter and a chain of single-pass filters.
        filters.forEach((filter) => { filter.padding = 0; });
        renderer.render({ container: scene, clear: true });
        context.drawImage(renderer.canvas, 0, 0);
        const pixels = context.getImageData(0, 0, width, height).data;

        assert(pixels.some((value, index) => index % 4 === 0 && value > 0 && value < 255), 'Expected a nonempty blur');

        scene.destroy({ children: true });

        return pixels;
    }

    const cases = [
        { strength: 8, quality: 1 },
        { strength: 8, quality: 2 },
        { strength: 8, quality: 3 },
        { strength: 8, quality: 4 },
        { strength: 8, quality: 4, clamp: true },
        { strength: 8, quality: 3, pixelSize: [2, 1] },
        { strength: 6, quality: 3, pixelSize: [1, 2], clamp: true },
    ];

    for (const options of cases)
    {
        const filter = new KawaseBlurFilter(options);
        const reference = filter.kernels.map((strength) => new KawaseBlurFilter({ ...options, strength, quality: 1 }));

        for (let frame = 0; frame < 3; frame++)
        {
            const actual = await render([filter]);
            const expected = await render(reference);
            const differing = actual.reduce((count, value, index) => count + (value !== expected[index]), 0);

            assert.equal(differing, 0, `${JSON.stringify(options)} frame ${frame}`);
        }
        filter.destroy();
        reference.forEach((pass) => pass.destroy());
    }
};
