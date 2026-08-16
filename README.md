# contrail-cirrus-detection

This repository contains my contributions to an Astro Pi Mission Space Lab investigation on the influence
of aircraft traffic on cirrus cloud formation. They have been organized, corrected, and documented.

### Research question

Do aircraft contribute to global warming by creating cirrus clouds?

### Approach

The main steps were:

- Identify the positions where aircraft intersected the trajectory of the ISS;
- Identify whether photographs were taken at those positions, after a certain interval from the passage of
  the aircraft;
- Check whether a cirrus cloud had formed and is visible in the image.

As part of the team, I created the CNN used to classify images and the functions to map geographical
coordinates to positions on those images. These have been collected in the two modules below, which have
applications beyond the scope of the project. Teammates handled sourcing the data and integrating all of
the code into a script containing the global logic of the approach, as well as the necessary additions
(e.g. API calls to the database, general I/O).

## Geolocation

Given
- a sequence of photographs, taken from orbit at known positions;
- a pair of lat/long coordinates identifying a location;

this module identifies where that location appears in a given image. The location need not actually fall
within the frame, as points outside the photograph are still assigned coordinates on the same plane.

The main problem solved is the arbitrary orientation of the camera with respect to the Earth. We require
that the camera points towards the nadir and that its orientation relative to its trajectory is fixed. We
then calculate two expressions of the same displacement:
- the distance in px between two consecutive photos, upwards and rightwards;
- the distance in km along the N and E directions, on the equirectangular projection of the surface.

These are sufficient to uniquely define the linear map between the two, which is a composition of a
rotation and a scaling. We implement it via complex multiplication, and use it to convert the km
displacement from the camera to the given coordinates into a px displacement in the image.

A fuller description is provided at the top of the file.

## Cirrus classifier

The network consists of three convolutional blocks, a dropout layer, and two dense layers. It has been
trained on ISS photographs, sorted by hand into `cirrus` and `not_cirrus`. The `kfold` (stratified 10-fold
cross validation), `train`, and `classify` commands can be used.

A number of challenges were encountered:
- **A severely imbalanced initial dataset**, which led the model to place every photo in the majority
  class. More photographs were selected to correct it, and stratified cross validation was used to preserve
  the balance across folds.
- **Overfitting**, addressed by modifying the network's structure. The choice of architecture was informed
  by k-fold cross validation, which was implemented to assess the model's capacity to generalize.
- **Resource waste** due to folds not releasing the memory they used. Solved by running each in a separate
  process, since a process releases all its memory, including GPU, on exit. This was possible in our case
  because the WSL VM uses the `fork` start method.

Training and validation accuracy were both approximately 84%. Also due to the code having run in WSL, the
initial dataset has since been lost, so this figure is not currently reproducible and the sample count and
class balance are unknown. A trained model and revised figures will be added once the dataset is
reconstructed.

## Revisions

On review, a number of apparent issues were found and corrected in both files. These are limited in scope
by the absence of the dataset, and are marked in `cirrus_classifier.py` with `FIX` comments. Among the more
notable are:

- **A missing factor of `cos(lat)`** in the calculation of km displacements along the N and E directions.
  This elongated only distances along latitude lines.
- **A corrected sign convention for `calculate_displacement`.** It now gives the displacement of the camera
  between the two photographs, rather than the displacement of the scene.
- **A class-label system no longer dependent on directory ordering.** `os.listdir` does not guarantee a
  particular order, so labels could be assigned to classes inconsistently between machines or runs. The
  names are now sorted and saved alongside the model, as opposed to inferred.
- **A reliable prediction-to-file pairing in `classify`.** It previously listed the sample directory twice
  and indexed predictions by the second listing, so any disagreement between the two listings paired a
  prediction with the wrong file.

Other minor improvements relating to presentation and reproducibility are documented in the files.

## Known limitations

- No data augmentation.
- The `Flatten` before the dense layer accounts for the large majority of the model's parameters
  (approx. 44.5M). This may be significantly improved by reducing to a 2D feature map instead (e.g. via
  global average pooling). Effectiveness will be judged upon testing with the recovered dataset.

## Running the files

`photo_geolocation.py` requires `opencv-python` and `numpy`. `cirrus_classifier.py` additionally requires
`tensorflow`, `scikit-learn`, `pillow`, and `matplotlib`; see `requirements.txt`. A `training_data/`
directory should also be provided, with one subdirectory per class.

Python 3.9–3.11 is required, as per the requirements of TensorFlow 2.13.1 and the type hints.

The photographs were originally taken from the
[Raspberry Pi Foundation's Flickr albums](https://www.flickr.com/photos/raspberrypi/albums), and are not
redistributed here.
