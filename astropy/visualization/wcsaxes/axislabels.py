# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import warnings
from collections.abc import Callable
from typing import Literal

import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import _api, rcParams
from matplotlib.text import Text

from astropy.utils.exceptions import AstropyUserWarning

from .frame import BaseFrame, RectangularFrame

LocLiteral = Literal["center", "left", "right", "bottom", "top"] | None


class AxisLabels(Text):
    __defaults_of_visual_properties_with_per_axis_dict = {"minpad": 1, "loc": None}

    def __init__(self, frame, minpad=1, *args, loc=None, **kwargs):
        # Use rcParams if the following parameters were not specified explicitly
        if "weight" not in kwargs:
            kwargs["weight"] = rcParams["axes.labelweight"]
        if "size" not in kwargs:
            kwargs["size"] = rcParams["axes.labelsize"]
        if "color" not in kwargs:
            kwargs["color"] = rcParams["axes.labelcolor"]
        if "verticalalignment" not in kwargs and "va" not in kwargs:
            kwargs["verticalalignment"] = "center"

        self._frame: BaseFrame = frame
        super().__init__(*args, **kwargs)
        self.set(
            minpad=minpad,
            loc=loc,
            clip_on=True,
            visible_axes="all",
            rotation_mode="anchor",
            visibility_rule="labels",
        )

    @staticmethod
    def _decorate_dict_visual_property_getter(method: Callable) -> Callable:
        @functools.wraps(method)
        def getter(self: "AxisLabels", axis: str) -> bool | int | float | str:
            property_value_or_dict = method(self)
            if isinstance(property_value_or_dict, dict):
                return property_value_or_dict[axis]
            else:
                return property_value_or_dict

        return getter

    @_decorate_dict_visual_property_getter
    def get_minpad(self):
        return self._minpad

    @_decorate_dict_visual_property_getter
    def get_loc(self) -> LocLiteral:
        return self._loc

    def set_visible_axes(self, visible_axes):
        self._visible_axes = self._frame._validate_positions(visible_axes)
        # re-verify the status of visual properties with dictionaries
        for (
            visual_property_name
        ) in self.__defaults_of_visual_properties_with_per_axis_dict.keys():
            visual_property_attribute = f"_{visual_property_name}"
            if hasattr(self, visual_property_attribute):
                visual_property_value_or_dict = getattr(self, visual_property_attribute)
                if isinstance(visual_property_value_or_dict, dict):
                    self._check_visual_property_dict_keys(
                        visual_property_value_or_dict, visual_property_name
                    )

    def get_visible_axes(self):
        if self._visible_axes == "all":
            return list(self._frame.keys())
        else:
            return [x for x in self._visible_axes if x in self._frame or x == "#"]

    def _check_visual_property_dict_keys(self, property_dict: dict, property_name: str):
        missing = set(self.get_visible_axes()).difference(property_dict.keys())
        if missing:
            q = "'"
            singular = len(missing) == 1
            warnings.warn(
                f"The configuration dictionary {property_dict} provided for "
                f"property {property_name} does not cover "
                f"{'axis' if singular else 'axes'} {
                    q + next(iter(missing)) + q if singular else missing
                } that "
                f"{'is' if singular else 'are'} currently set to be visible; "
                "drawing may raise a KeyError for this property.",
                AstropyUserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _decorate_dict_visual_property_setter(method: Callable) -> Callable:
        property_name = method.__name__.split("set_")[1]

        @functools.wraps(method)
        def setter(
            self: "AxisLabels", property_value_or_dict: bool | float | str | dict
        ) -> None:
            if isinstance(property_value_or_dict, dict):
                self._check_visual_property_dict_keys(
                    property_value_or_dict, property_name
                )
            method(self, property_value_or_dict)

        return setter

    @_decorate_dict_visual_property_setter
    def set_minpad(self, minpad):
        self._minpad = minpad

    @_decorate_dict_visual_property_setter
    def set_loc(self, loc: LocLiteral) -> None:
        self._loc = loc

    def set_visibility_rule(self, value):
        allowed = ["always", "labels", "ticks"]
        if value not in allowed:
            raise ValueError(
                f"Axis label visibility rule must be one of {' / '.join(allowed)}"
            )

        self._visibility_rule = value

    def get_visibility_rule(self):
        return self._visibility_rule

    def draw(
        self,
        renderer,
        bboxes,
        ticklabels_bbox,
        coord_ticklabels_bbox,
        ticks_locs,
        visible_ticks,
    ):
        if not self.get_visible():
            return

        text_size = renderer.points_to_pixels(self.get_size())
        # Flatten the bboxes for all coords and all axes
        ticklabels_bbox_list = []
        for bbcoord in ticklabels_bbox.values():
            for bbaxis in bbcoord.values():
                ticklabels_bbox_list += bbaxis

        for axis in self.get_visible_axes():
            if axis == "#":
                continue

            if self.get_visibility_rule() == "ticks":
                if not ticks_locs[axis]:
                    continue
            elif self.get_visibility_rule() == "labels":
                if not coord_ticklabels_bbox:
                    continue

            padding = text_size * self.get_minpad(axis)

            loc = self.get_loc(axis)
            if axis in {"t", "b", "h"}:
                loc = loc if loc is not None else rcParams["xaxis.labellocation"]
                _api.check_in_list(("left", "center", "right"), loc=loc)

                bary = {
                    "left": 0,
                    "center": 0.5,
                    "right": 1,
                }[loc]
            elif axis in {"l", "r", "v"}:
                loc = loc if loc is not None else rcParams["yaxis.labellocation"]
                _api.check_in_list(("bottom", "center", "top"), loc=loc)

                bary, loc = {
                    "bottom": (0, "right"),
                    "center": (0.5, "center"),
                    "top": (1, "left"),
                }[loc]
            elif loc is not None and loc != "center":
                raise NotImplementedError(
                    f"Received unsupported value {loc=!r}. "
                    f"Only loc='center' is implemented for {axis=!r}"
                )
            else:
                loc = "center"
                bary = 0.5

            # Find position of the axis label. For now we pick the mid-point
            # along the path but in future we could allow this to be a
            # parameter.
            x, y, normal_angle = self._frame[axis]._barycentric_x_y_angle(bary)

            label_angle = (normal_angle - 90.0) % 360.0
            if 135 < label_angle < 225:
                label_angle += 180
            self.set_rotation(label_angle)
            if 45 < label_angle < 135:
                match loc:
                    case "left":
                        loc = "right"
                    case "right":
                        loc = "left"
            self.set_ha(loc)

            # Find label position by looking at the bounding box of ticks'
            # labels and the image. It sets the default padding at 1 times the
            # axis label font size which can also be changed by setting
            # the minpad parameter.

            if isinstance(self._frame, RectangularFrame):
                if (
                    len(ticklabels_bbox_list) > 0
                    and ticklabels_bbox_list[0] is not None
                ):
                    coord_ticklabels_bbox[axis] = [
                        mtransforms.Bbox.union(ticklabels_bbox_list)
                    ]
                else:
                    coord_ticklabels_bbox[axis] = [None]

                visible = (
                    axis in visible_ticks and coord_ticklabels_bbox[axis][0] is not None
                )

                if axis == "l":
                    if visible:
                        x = coord_ticklabels_bbox[axis][0].xmin
                    x = x - padding

                elif axis == "r":
                    if visible:
                        x = coord_ticklabels_bbox[axis][0].x1
                    x = x + padding

                elif axis == "b":
                    if visible:
                        y = coord_ticklabels_bbox[axis][0].ymin
                    y = y - padding

                elif axis == "t":
                    if visible:
                        y = coord_ticklabels_bbox[axis][0].y1
                    y = y + padding

            else:  # arbitrary axis
                x = x + np.cos(np.radians(normal_angle)) * (padding + text_size * 1.5)
                y = y + np.sin(np.radians(normal_angle)) * (padding + text_size * 1.5)

            self.set_position((x, y))
            super().draw(renderer)

            bb = super().get_window_extent(renderer)
            bboxes.append(bb)
