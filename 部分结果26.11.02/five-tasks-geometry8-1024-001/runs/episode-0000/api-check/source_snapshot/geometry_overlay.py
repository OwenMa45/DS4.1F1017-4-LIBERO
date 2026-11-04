"""Calibration-only visual audit; never changes the policy's input images."""
import io
import math


def draw_geometry_overlay(obs, camera):
    from PIL import Image, ImageDraw
    source = Image.open(io.BytesIO(obs.images[camera])).convert("RGB")
    width, height = source.size
    output = Image.new("RGB", (width * 2, height * 2 + 34), "white")
    output.paste(source.resize((width * 2, height * 2)), (0, 0))
    draw = ImageDraw.Draw(output)
    geometry = obs.geometry
    points = {"E": geometry["cameras"][camera]["eef_pixel_uv"]}
    pads = geometry["robot_gripper_geometry"]["pixels_uv"][camera]
    points.update({"L": pads["left_pad"], "R": pads["right_pad"], "M": pads["pad_midpoint"]})
    colors = {"E": "cyan", "L": "yellow", "R": "magenta", "M": "lime"}
    hidden = []
    for key, point in points.items():
        if (point is None or not all(math.isfinite(x) for x in point)
                or not 0 <= point[0] < width or not 0 <= point[1] < height):
            hidden.append(key)
            continue
        x, y = (round(2 * v) for v in point)
        draw.line((x-5, y, x+5, y), fill=colors[key], width=2)
        draw.line((x, y-5, x, y+5), fill=colors[key], width=2)
        # Offset nearby E/M labels; the cross itself remains at the projection.
        draw.text((max(0, x-20) if key == "E" else x+7, y-13 if key == "M" else y+3),
                  key, fill=colors[key], stroke_width=1, stroke_fill="black")
    draw.text((5, height*2+2), "E: control site   L/R: pad centres   M: midpoint", fill="black")
    draw.text((5, height*2+17), "Off-screen/behind: " + (", ".join(hidden) or "none"), fill="black")
    data = io.BytesIO()
    output.save(data, format="PNG")
    return data.getvalue()
