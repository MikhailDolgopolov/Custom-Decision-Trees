import colorsys


def save_dot_to_file(dot_data, file_path='iris_tree'):
    with open(file_path, 'w') as file:
        file.write(dot_data)

def generate_distinct_colors_hex(n):
    return [f'#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}'
            for i in range(n)
            for r, g, b in [colorsys.hsv_to_rgb(i / n, 1, 1)]]