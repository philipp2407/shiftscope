import colorsys
import matplotlib.colors as clrs
import numpy as np


CellfaceStdNorm = clrs.Normalize(vmin=-4, vmax=14, clip=True)
CellfaceStdCMap = clrs.LinearSegmentedColormap.from_list(
    "CellFace Standard",
    [
        # Position               R     G     B     A
        (CellfaceStdNorm(-4.0), [0.65, 0.93, 1.00, 1.0]),  # Air bubbles
        (CellfaceStdNorm(0.0), [1.00, 0.97, 0.96, 1.0]),  # Background
    ]
    + [
        (
            CellfaceStdNorm(2 + p * (14 - 2)),
            list(
                colorsys.hsv_to_rgb(
                    (265 + 150 * p) / 360,  # Hue: From Blue to Purple
                    0.7 + 0.3 * p,  # Saturation: Pastel to fully saturated
                    (1 - p) ** 1.5,  # Value: From Bright to Black
                )
            )
            + [1.0],
        )
        for p in np.linspace(0.0, 1.0, 20)
    ],
)


# Colormap adjusted
CellfaceStdCMapLeukocytes = clrs.LinearSegmentedColormap.from_list(
    "CellFace Standard Leukocytes",
    [
        # Position               R     G     B     A
        (CellfaceStdNorm(-4.0), [0.65, 0.93, 1.00, 1.0]),  # Air bubbles
        (CellfaceStdNorm(0.0), [1.00, 0.97, 0.96, 1.0]),  # Background
    ]
    + [
        (
            CellfaceStdNorm(2 + p * (14 - 2)),
            list(
                colorsys.hsv_to_rgb(
                    (280 - 90 * p) / 360,  # Hue: From Blue to Purple
                    0.5 + 1 * p,  # Saturation: Pastel to fully saturated
                    1 - (1+np.exp(-2*(-4+8*(p+.27))))**(-1),  # Value: From Bright to Black
                )
            )
            + [1.0],
        )
        for p in np.linspace(0.0, 1.0, 20)
    ],
)