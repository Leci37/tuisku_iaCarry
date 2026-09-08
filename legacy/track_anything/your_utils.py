import numpy as np
import cv2


def parse_label_map(pbtxt_path):
    label_map = {}
    with open(pbtxt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    items = content.split('item {')
    for item in items[1:]:
        fields = {
            "id": None,
            "name": None,
            "hex_color": None,
            "description": None
        }
        lines = item.strip().splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("id:"):
                fields["id"] = int(line.split(":")[1].strip())
            elif line.startswith("name:"):
                fields["name"] = line.split(":")[1].strip().strip('"')
            elif line.startswith("hex_color:"):
                fields["hex_color"] = line.split(":")[1].strip().strip('"')
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                fields["description"] = desc.strip('"')

        if fields["id"] is not None:
            label_map[fields["id"]] = fields

    return label_map


def overlay_mask(image, mask, label_color_map=None, label_ids=None):
    color_mask = np.zeros_like(image)

    for uid in np.unique(mask):
        if uid == 0:
            continue

        if label_color_map and label_ids:
            label_name = label_ids.get(uid, None)
            if label_name and label_name in label_color_map:
                hex_color = label_color_map[label_name]
                color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color = color[::-1]  # RGB to BGR for OpenCV
            else:
                color = np.random.randint(0, 255, size=3)
        else:
            color = np.random.randint(0, 255, size=3)

        color_mask[mask == uid] = color

    return cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)


def build_label_ui_data(label_map):
    """
    Build display HTML, color map, and reverse lookup for label names.
    Returns:
        label_display: list of HTML-rendered label names with color
        label_html_to_name: dict mapping HTML to label names
        label_color_map: dict mapping label names to their hex colors
    """
    label_display = []
    label_html_to_name = {}
    label_color_map = {}

    for entry in label_map.values():
        name = entry["name"]
        color = entry["hex_color"]
        label_color_map[name] = color

        html = (
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 15px; height: 15px; background: {color}; "
            f"border: 1px solid #000; margin-right: 6px;'></div>"
            f"<span>{name}</span></div>"
        )

        label_display.append(html)
        label_html_to_name[html] = name

    return label_display, label_html_to_name, label_color_map


def make_label_html(name, color):
    return f"""
    <div style='display: flex; align-items: center; gap: 8px;'>
        <div style='width: 14px; height: 14px; background-color: {color}; border: 1px solid #000;'></div>
        <span style='font-size: 14px;'>{name}</span>
    </div>
    """




def select_label(label, state,label_color_map):
    state["selected_label"] = label
    hex_color = label_color_map.get(label, "#FFFFFF")
    html_box = f"<div style='width:30px;height:20px;background:{hex_color};border:1px solid #000;'></div>"
    return f"Selected: {label}", html_box

def format_label(name):
    name = name.replace("_", " ").capitalize()
    accents = {
        "aguila": "AGUIla", "axedrak": "AXEdrak", "chipsahoy": "CHIPsahoy",
        "cocacola": "cocACOla", "colacao": "colACola", "colgate": "colGAte",
        "estrellag": "estRELLAg", "fanta": "FANta", "hysori": "hysORI",
        "mahou00": "maHOu 00", "mahou5": "maHOu 5", "smacks": "SMAcks",
        "tostarica": "tosTArica", "asturiana": "astuRIANa"
    }
    base = name.split(" ")[0].lower()
    return accents.get(base, name.replace("_", " ").capitalize()) +" "+name.split(" ")[1].lower()


def build_legend_html(label_map):
    entries = list(label_map.values())
    half = (len(entries) + 1) // 2  # Round up for left column

    left_entries = entries[:half]
    right_entries = entries[half:]

    def make_column(entries):
        html = "<div style='display: flex; flex-direction: column; gap: 6px;'>"
        for entry in entries:
            name = entry["name"]
            display_name = format_label(name)
            color = entry["hex_color"]
            html += (
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 21px; height: 16px; background: {color}; border: 1px solid #000; margin-right: 8px;'></div>"
                f"<span style='font-size: 19px;'>{display_name}</span></div>"
            )
        html += "</div>"
        return html

    left_col = make_column(left_entries)
    right_col = make_column(right_entries)

    return (
        "<div style='display: flex; flex-direction: row; gap: 40px;'>"
        f"{left_col}{right_col}</div>"
    )


# def format_label(name):
#     name = name.replace("_", " ").capitalize()
#     # Example: mark accented syllables manually
#     accents = {
#         "aguila": "AGUIla",
#         "axedrak": "AXEdrak",
#         "chipsahoy": "CHIPsahoy",
#         "cocacola": "cocACOla",
#         "colacao": "colACola",
#         "colgate": "colGAte",
#         "estrellag": "estRELLAg",
#         "fanta": "FANta",
#         "hysori": "hysORI",
#         "mahou00": "maHOu 00",
#         "mahou5": "maHOu 5",
#         "smacks": "SMAcks",
#         "tostarica": "tosTArica",
#         "asturiana": "astuRIANa"
#     }
#     base = name.split()[0].lower()
#     return accents.get(base, name)

