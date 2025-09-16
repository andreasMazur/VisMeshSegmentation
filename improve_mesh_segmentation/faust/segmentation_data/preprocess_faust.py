from geoconv_examples.mpi_faust.data.preprocess_faust import preprocess_faust


def get_faust_dataset(target_dir, registration_path, geodesic_diameters_path):
    """Preprocess the FAUST dataset for segmentation tasks.

    Parameters
    ----------
    target_dir: str
        The directory where the preprocessed dataset will be saved.
    registration_path: str
        The path to the directory containing the registered FAUST meshes.
    geodesic_diameters_path: str
        The path to the numpy file containing precomputed geodesic diameters.
    """
    preprocess_faust(
        n_radial=3,
        n_angular=6,
        target_dir=target_dir,
        registration_path=registration_path,
        shot=True,
        geodesic_diameters_path=geodesic_diameters_path,
        precomputed_gpc_radius=-1.,
        processes=10,
        add_noise=False
    )
