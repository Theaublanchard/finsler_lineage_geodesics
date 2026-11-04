"""
Interactive vector selection widget for matplotlib plots.
Requires ipympl backend: %matplotlib widget
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch

# Optional ipywidgets import for Output widget support
try:
    from ipywidgets import Output
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False


class VectorSelector:
    """
    Interactive click-and-drag vector selector for matplotlib axes.

    Usage:
        # In Jupyter notebook, first enable widget backend:
        # %matplotlib widget

        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1])

        def my_callback(x, v):
            # x: numpy array of shape (2,) - starting point in data coords
            # v: numpy array of shape (2,) - vector from start to end
            result = model.F(torch.tensor(x), torch.tensor(v))
            print(f"F({x}, {v}) = {result}")

        selector = VectorSelector(ax, callback_fn=my_callback)
        plt.show()

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to attach the selector to.
    callback_fn : callable
        Function to call on mouse release. Signature: callback_fn(x, v)
        where x is the start point and v is the vector (both numpy arrays).
        Can optionally return a string to display as result.
    arrow_color : str, optional
        Color of the arrow during and after drag. Default: 'red'.
    arrow_width : float, optional
        Width of the arrow. Default: 2.0.
    persist_arrow : bool, optional
        If True, keep the arrow visible after release. Default: False.
    show_coordinates : bool, optional
        If True, display coordinates as text annotation. Default: True.
    output_widget : ipywidgets.Output, optional
        If provided, print statements will be captured and displayed in this widget.
        Create with: `from ipywidgets import Output; out = Output()`
    """

    def __init__(
        self,
        ax,
        callback_fn,
        arrow_color='red',
        arrow_width=2.0,
        persist_arrow=False,
        show_coordinates=True,
        output_widget=None,
    ):
        self.ax = ax
        self.fig = ax.figure
        self.callback_fn = callback_fn
        self.arrow_color = arrow_color
        self.arrow_width = arrow_width
        self.persist_arrow = persist_arrow
        self.show_coordinates = show_coordinates
        self.output_widget = output_widget

        # State variables
        self.start_point = None  # (x, y) in data coordinates
        self.arrow = None        # FancyArrowPatch artist
        self.text_annotation = None
        self.result_annotation = None  # For displaying callback results
        self.is_dragging = False

        # Connect event handlers
        self._cid_press = self.fig.canvas.mpl_connect(
            'button_press_event', self._on_press
        )
        self._cid_motion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self._on_motion
        )
        self._cid_release = self.fig.canvas.mpl_connect(
            'button_release_event', self._on_release
        )

    def _on_press(self, event):
        """Handle mouse button press."""
        # Ignore clicks outside the axes
        if event.inaxes != self.ax:
            return

        # Only respond to left mouse button
        if event.button != 1:
            return

        # Store starting point in DATA coordinates
        self.start_point = np.array([event.xdata, event.ydata])
        self.is_dragging = True

        # Clear previous arrow if not persisting
        self._clear_artists()

        # Create arrow starting at the click point (zero length initially)
        self.arrow = FancyArrowPatch(
            posA=self.start_point,
            posB=self.start_point,
            arrowstyle='-|>',
            color=self.arrow_color,
            linewidth=self.arrow_width,
            mutation_scale=15,
        )
        self.ax.add_patch(self.arrow)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        """Handle mouse motion during drag."""
        if not self.is_dragging:
            return

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Update arrow endpoint
        end_point = np.array([event.xdata, event.ydata])

        # Update arrow
        self.arrow.set_positions(self.start_point, end_point)

        # Update text annotation if enabled
        if self.show_coordinates:
            v = end_point - self.start_point
            self._update_annotation(self.start_point, v)

        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """Handle mouse button release."""
        if not self.is_dragging:
            return

        self.is_dragging = False

        # Get final endpoint (use last valid position if outside axes)
        if event.xdata is not None and event.ydata is not None:
            end_point = np.array([event.xdata, event.ydata])
        else:
            # Use the arrow's current end position
            end_point = np.array(self.arrow.get_path().vertices[-1])

        # Calculate vector
        v = end_point - self.start_point

        # Check for zero-length vector
        if np.linalg.norm(v) < 1e-10:
            self._clear_artists()
            return

        # Call the callback function
        if self.callback_fn is not None:
            result = None
            if self.output_widget is not None:
                # Capture output in the widget
                with self.output_widget:
                    self.output_widget.clear_output(wait=True)
                    result = self.callback_fn(self.start_point.copy(), v.copy())
            else:
                result = self.callback_fn(self.start_point.copy(), v.copy())

            # If callback returns a string, display it on the plot
            if result is not None and isinstance(result, str):
                self._show_result(result)

        # Clear arrow if not persisting
        if not self.persist_arrow:
            self._clear_artists()

        self.fig.canvas.draw_idle()

    def _update_annotation(self, x, v):
        """Update or create text annotation showing coordinates."""
        text = f"x=({x[0]:.2f}, {x[1]:.2f})\nv=({v[0]:.2f}, {v[1]:.2f})"

        if self.text_annotation is None:
            self.text_annotation = self.ax.annotate(
                text,
                xy=x,
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            )
        else:
            self.text_annotation.set_text(text)
            self.text_annotation.xy = x

    def _show_result(self, text):
        """Display result text on the plot (top-left corner)."""
        # Remove previous result annotation
        if self.result_annotation is not None:
            self.result_annotation.remove()
            self.result_annotation = None

        self.result_annotation = self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
        )

    def _clear_artists(self):
        """Remove arrow and annotation from axes."""
        if self.arrow is not None:
            self.arrow.remove()
            self.arrow = None

        if self.text_annotation is not None:
            self.text_annotation.remove()
            self.text_annotation = None

        # Note: result_annotation persists until next callback

    def disconnect(self):
        """Disconnect all event handlers and clean up."""
        self._clear_artists()
        self.fig.canvas.mpl_disconnect(self._cid_press)
        self.fig.canvas.mpl_disconnect(self._cid_motion)
        self.fig.canvas.mpl_disconnect(self._cid_release)
