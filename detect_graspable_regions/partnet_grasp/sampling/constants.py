# https://github.com/daerduoCarey/partnet_seg_exps/tree/master/stats/train_val_test_split
PARTNET_EXP_GITHUB_LINK = (
    "https://raw.githubusercontent.com/daerduoCarey/partnet_seg_exps/master/stats/train_val_test_split/"
)

# (Incomplete) mapping from wordnet synsets to ShapeNet categories
CAT2SYNSET = dict(
    mug="03797390",
    bag="02773838",
    bottle="02876657",
    can="02946921",
    vessel="04530566",
)

LABLEMAP = dict(
    body=1,
    handle=2,
    containing_things=4,
    other=4,
)

TMPMAP = {
    0.: 'unknown',
    1.: 'body',
    2.: 'handle',
    4.: 'containing_things',
    3.: 'unknown',
    5.: 'unknown'
}

COLORMAP = dict(
    unknown=[255, 0, 0, 255],
    handle=[0, 255, 0, 255],
    body=[255, 0, 0, 255],
    containing_things=[213, 200, 48, 255],
    other=[213, 200, 48, 255],
)

USER_COLORS = [
    [0, 255, 0, 255],
    [255, 127, 0, 255],
    [119, 221, 231, 255],
]
