from typing import Union, Any
import io
import base64

from matplotlib.pyplot import Figure


def mpl_to_base64(fig_or_ax: Union[Figure, Any], dpi: int = 80, as_data_uri: bool = False) -> str:
    """
    Converts a matplotlib figure to a base64 encoded string.

    Args:
        fig (plt.Figure): The figure to convert.

    Returns:
        str: The base64 encoded string.

    """
    # get the figure
    if isinstance(fig_or_ax, Figure):
        fig = fig_or_ax
    elif not hasattr(fig_or_ax, 'figure'):
        raise TypeError("fig_or_ax must be a matplotlib figure or an object with a figure attribute resolving to one.")
    else:
        fig = fig_or_ax.figure
    
    # create a buffer and save the figure
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi)
    
    # create base64 encoded string
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    if as_data_uri:
        return f"data:image/png;base64,{data}"
    else:
        return data
