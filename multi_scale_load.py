import napari

from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from _generative_yt.yt_store import yt_dataset, random_np_generative_ds


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer(ndisplay=3)

    multiscale_img = yt_dataset(max_levels=10)
    # multiscale_img = random_np_generative_ds()
    print(multiscale_img[0])

    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=[0, 1],
        colormap='viridis',
        ndisplay=3,
        rendering='mip',
    )

    viewer.axes.visible = True
    napari.run()
