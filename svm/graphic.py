import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

_STEP_COUNT = 380
POSITIVE_POINT_COLOR = 'red'
NEGATIVE_POINT_COLOR = 'blue'
POSITIVE_AREA_COLOR = 'lightcoral'
NEGATIVE_AREA_COLOR = 'cornflowerblue'


def draw_graphic(dataset, targets, predictor, title, fname=None):
    x = dataset[:, 0]
    y = dataset[:, 1]
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    x_step = (max_x - min_x) / _STEP_COUNT
    y_step = (max_y - min_y) / _STEP_COUNT
    x_grid, y_grid = np.meshgrid(np.arange(min_x, max_x, x_step), np.arange(min_y, max_y, y_step))
    rows_count = len(x_grid)
    columns_count = len(x_grid[0])
    predictions = np.ndarray((rows_count, columns_count))
    for i in range(rows_count):
        for j in range(columns_count):
            predictions[i][j] = _predict(predictor, x_grid[i][j], y_grid[i][j])

    colorMap = colors.ListedColormap([NEGATIVE_AREA_COLOR, POSITIVE_AREA_COLOR])
    plt.pcolormesh(x_grid, y_grid, predictions, cmap=colorMap, shading='auto')
    _draw_data_points(dataset, targets)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)
    plt.show()


def _predict(predictor, x, y):
    return predictor.predict(np.array([x, y]))


def _draw_data_points(dataset, targets):
    positive_x = np.array([])
    positive_y = np.array([])
    negative_x = np.array([])
    negative_y = np.array([])
    for i in range(len(targets)):
        if targets[i] == 1:
            positive_x = np.append(positive_x, dataset[i][0])
            positive_y = np.append(positive_y, dataset[i][1])
        else:
            negative_x = np.append(negative_x, dataset[i][0])
            negative_y = np.append(negative_y, dataset[i][1])
    plt.scatter(positive_x, positive_y, c=POSITIVE_POINT_COLOR, marker='o')
    plt.scatter(negative_x, negative_y, c=NEGATIVE_POINT_COLOR, marker='o')
