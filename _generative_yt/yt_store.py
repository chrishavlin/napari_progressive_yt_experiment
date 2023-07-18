from napari.experimental._generative_zarr import MandelbulbStore, tile_bounds, create_meta_store
import yt
import dask.array as da
import numpy as np
import zarr

class RandomNpStore(MandelbulbStore):
    def get_chunk(self, level, z, y, x):

        tile = self._fetch_data()  # this is where the mandelbulb call was
        tile = tile.transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile

    def _fetch_data(self):
        data = np.random.randint(0, 255,
                                 (self.tilesize, self.tilesize, self.tilesize),
                                 dtype=self.dtype)
        return data


class YTStore(MandelbulbStore):

    def __init__(
        self, yt_file, yt_field, levels, tilesize, maxiter=255,
            compressor=None, order=4,
            use_yt_load_sample=False,
            take_log=True,
    ):

        # need to copy/pase rather than override cause of the d-type handling here
        self.levels = levels
        self.tilesize = tilesize
        self.compressor = compressor
        self.dtype = np.float32
        self.maxiter = maxiter
        self.order = order
        self._store = create_meta_store(
            levels, tilesize, compressor, self.dtype, ndim=3
        )

        if use_yt_load_sample:
            self.ds = yt.load_sample(yt_file)
        else:
            self.ds = yt.load(yt_file)
        
        self.ds.index  # make sure the index is built on initialization            
        self.field = yt_field 
        self.max_coord = 1.0
        self.min_coord = 0.0
        self.take_log = take_log
        self.data_min, self.data_max = self.find_field_range()

    def find_field_range(self):
        ad = self.ds.all_data()
        min_field, max_field = ad.quantities.extrema(self.field)
        min_field = min_field.d
        max_field = max_field.d
        if self.take_log:
            min_field = np.log10(min_field)
            max_field = np.log10(max_field)
        return min_field, max_field


    def get_chunk(self, level, z, y, x):
        # z, y, x are tile centers        
        bounds = tile_bounds(level, (x, y, z), 
                             self.levels,                              
                             min_coord=self.min_coord,
                             max_coord=self.max_coord, )  
        
        tile = self._fetch_data(bounds)  # this is where the mandelbulb call was        

        # tile = tile.transpose()
        # tile = tile.reshape(
        #     self.tilesize, self.tilesize, self.tilesize
        # ).transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile

    def _fetch_data(self, bounds):        
        LE = self.ds.arr(bounds[:, 0], "code_length")
        RE = self.ds.arr(bounds[:, 1], "code_length")
        frb = self.ds.r[LE[0] : RE[0] : complex(0, self.tilesize),
                        LE[1] : RE[1] : complex(0, self.tilesize),
                        LE[2] : RE[2] : complex(0, self.tilesize)
        ]
        data = frb[self.field]
        if self.take_log:
            data = np.log10(data)

        data = (data - self.data_min) / (self.data_max - self.data_min)
        # data = np.random.random((self.tilesize, self.tilesize, self.tilesize))
        return data.astype(self.dtype)


def random_np_generative_ds(max_levels=10):
    chunk_size = (32, 32, 32)

    # Initialize the store
    # yt_store = YTStore("IsolatedGalaxy", ("enzo", "Density"), 6, 32, use_yt_load_sample=True, take_log=False)
    store = zarr.storage.KVStore(
        RandomNpStore(
                max_levels,
                32,
        )
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    # store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for _scale, a in enumerate(multiscale_img):
        da.core.normalize_chunks(
            chunk_size,
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        arrays += [a]

    return arrays

def yt_dataset(max_levels=8):
    """Generate a multiscale image of the yt dataset set for a given number
    of levels/scales. Scale 0 will be the highest resolution.

    This is intended to be used with progressive loading. As such, it returns
    a dictionary will all the metadata required to load as multiple scaled
    image layers via add_progressive_loading_image

    >>> large_image = yt_dataset(max_levels=14)
    >>> multiscale_img = large_image["arrays"]
    >>> viewer._layer_slicer._force_sync = False
    >>> add_progressive_loading_image(multiscale_img, viewer=viewer)

    Parameters
    ----------
    max_levels: int
        Maximum number of levels (scales) to generate

    Returns
    -------
    Dictionary of multiscale data with keys ['container', 'dataset',
        'scale levels', 'scale_factors', 'chunk_size', 'arrays']
    """
    chunk_size = (32, 32, 32)

    # Initialize the store
    # yt_store = YTStore("IsolatedGalaxy", ("enzo", "Density"), 6, 32, use_yt_load_sample=True, take_log=False)    
    store = zarr.storage.KVStore(
        YTStore("IsolatedGalaxy", 
                ("enzo", "Density"), 
                max_levels, 
                32, 
                use_yt_load_sample=True, 
                take_log=True)
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    # store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for _scale, a in enumerate(multiscale_img):
        da.core.normalize_chunks(
            chunk_size,
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        arrays += [a]

    return arrays