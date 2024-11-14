import colorsys
import os

from graphviz import Source


def render_partial_dot(dot_str, filename="partial_graph", format="png"):
    """
    Renders a partial Graphviz DOT graph by adding a closing brace '}' if not already present.

    Parameters:
    - dot_str (str): The DOT string representing the partial graph.
    - filename (str): The filename to save the rendered image.
    - format (str): The format of the output file (e.g., 'png', 'pdf').

    Returns:
    - graphviz.Source: A Source object for the rendered graph.
    """
    # Check if the dot string ends with '}' and add it if missing
    if not dot_str.strip().endswith("}"):
        dot_str += "\n}"

    filename = f"{filename}.{format}"
    count = 1
    while os.path.exists(filename):
        filename = f"{filename}_{count}.{format}"
        count += 1

    # Create a Graphviz source object and render
    graph = Source(dot_str)
    graph.format = format
    graph.render(filename=filename.replace(f".{format}", ""), view=True)  # Automatically view the file after rendering
    return graph


def save_dot_to_file(dot_data, file_path='iris_tree'):
    with open(file_path, 'w') as file:
        file.write(dot_data)

def generate_distinct_colors_hex(n):
    return [f'#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}'
            for i in range(n)
            for r, g, b in [colorsys.hsv_to_rgb(i / n, 1, 1)]]