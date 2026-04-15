import numpy as np
from .dataclasses import EnvironmentSettings

def generate_detachment_wall_layout(env_settings: EnvironmentSettings, walkable_width: int = 3) -> np.ndarray:
    width = env_settings.width
    height = env_settings.height
    shell = _generate_shell(width, height)
    middle_section = _generate_middle_section(width, height, walkable_width)
    spiral = _generate_spiral(width, height, walkable_width)

    layout = shell + middle_section + spiral
    return np.array(layout)

def _generate_shell(env_width: int, env_height: int) -> list[tuple[int,int]]: # TODO extend for variable shell width?
    top = [(col, 0) for col in range(env_width)]
    bottom = [(col, env_height-1) for col in range(env_width)]
    right = [(env_width-1, row) for row in range(env_height)]
    left = [(0, row) for row in range(env_height)]

    shell = top + right + bottom + left 
    return shell

def _generate_middle_section(env_width: int, env_height: int, walkable_width: int) -> list[tuple[int,int]]:
    best_middle_width = _calc_best_split(env_width)
    walkable_height_start = (env_height - walkable_width) // 2
    walkable_height_end = walkable_height_start + walkable_width
    middle_width_start = (env_width - best_middle_width) // 2
    middle_width_end = middle_width_start + best_middle_width

    top = [(col, row) for row in range(walkable_height_start) for col in range(middle_width_start, middle_width_end)]
    bottom = [(col, row) for row in range(walkable_height_end, env_height) for col in range(middle_width_start - walkable_width - 1, middle_width_end + walkable_width + 1)]

    return top + bottom

def _calc_best_split(env_width: int):
    best_middle_width = 0
    for col in range(1 , env_width//2):
        middle_width = env_width % (col)
        if (middle_width != 0 and middle_width >= best_middle_width):
            best_middle_width = middle_width
    return best_middle_width

def _generate_spiral(env_width: int, env_height: int, walkable_width: int) -> list[tuple[int,int]]:
    middle_width_start = _calc_best_split(env_width)
    bottom_split_width_start = middle_width_start + walkable_width + 1 

    middle_width_start = (env_width - middle_width_start) // 2
    middle_width_end = middle_width_start + middle_width_start 

    space_left = env_height - 2 # - 2 given the shell
    max_possible_layers = 0 # start at -1 as the first outer loop dosn't count
    while(space_left > 0):
        space_left -= walkable_width * 2
        if (space_left > 0):
            max_possible_layers += 1
    
    adjusted_env_height = env_height - 1 # - 1 given the shell
    spiral = []
    while (max_possible_layers > 0):
        left_walkable_offset = (walkable_width + 1) * (max_possible_layers) # + 1 for the wall thickness
        right_walkable_offset = left_walkable_offset
        layer_row_end = adjusted_env_height

        top_left =      [(col, left_walkable_offset) for col in range(left_walkable_offset, bottom_split_width_start - left_walkable_offset)]
        left_left =     [(left_walkable_offset, row) for row in range(left_walkable_offset, layer_row_end - left_walkable_offset)]
        right_left =    [(bottom_split_width_start - left_walkable_offset - 1, row) for row in range(left_walkable_offset, bottom_split_width_start - left_walkable_offset + walkable_width)]
        bottom_left =   [(col, adjusted_env_height - left_walkable_offset) for col in range(left_walkable_offset, bottom_split_width_start - left_walkable_offset - walkable_width - 1)]

        top_right =      [(col, right_walkable_offset) for col in range(middle_width_end + right_walkable_offset - walkable_width, env_width - 1 - right_walkable_offset)]
        left_right =     [(middle_width_end + right_walkable_offset - walkable_width - 1, row) for row in range(right_walkable_offset, bottom_split_width_start - right_walkable_offset + walkable_width)]
        right_right =    [(env_width - right_walkable_offset - 1, row) for row in range(right_walkable_offset, layer_row_end - right_walkable_offset)]
        bottom_right =   [(col, adjusted_env_height - right_walkable_offset) for col in range(middle_width_end + right_walkable_offset, env_width - right_walkable_offset)]

        spiral += top_left
        spiral += left_left 
        spiral += right_left
        spiral += bottom_left

        spiral += top_right
        spiral += left_right
        spiral += right_right
        spiral += bottom_right

        max_possible_layers -= 1

    return spiral

def generate_derailment_wall_layout(env_settings: EnvironmentSettings) -> list[tuple[int,int]]:
    width = env_settings.width
    height = env_settings.height
 
    shell = _generate_shell(width, height)
    maze = _generate_maze()

    return shell + maze

def _generate_maze() -> list[tuple[int,int]]:
    block1 = []
    for col in range(10, 19):
        for row in range(1,3):
            cor = (col, row)
            block1.append(cor)
    
    block2 = [(20,5), (20,6), (23,7), (24,7)]
    for col in range(21, 25):
        for row in range(4,7):
            cor = (col, row)
            block2.append(cor)

    block3 = []
    for col in range(17, 19):
        for row in range(6,8):
            cor = (col, row)
            block3.append(cor)

    block4 = []
    for col in range(17, 23):
        for row in range(9,11):
            cor = (col, row)
            block4.append(cor)

    block5 = []
    for col in range(25, 29):
        for row in range(11,15):
            cor = (col, row)
            block5.append(cor)
    for col in range(17, 25):
        for row in range(14,15):
            cor = (col, row)
            block5.append(cor)

    block6 = []
    for col in range(13, 16):
        for row in range(5,13):
            cor = (col, row)
            block6.append(cor)
    for col in range(1, 16):
        for row in range(7,9):
            cor = (col, row)
            block6.append(cor)
    for col in range(9, 16):
        for row in range(9,10):
            cor = (col, row)
            block6.append(cor)
   
    block7 = []
    for col in range(10, 12):
        for row in range(11,15):
            cor = (col, row)
            block7.append(cor)

    block8 = []
    for col in range(5, 9):
        for row in range(13,15):
            cor = (col, row)
            block8.append(cor)


    return block1 + block2 + block3 + block4 + block5 + block6 + block7 + block8

def _generate_square_room(left_corner: tuple[int, int], right_corner: tuple[int, int]) -> tuple[list[tuple[int,int]], tuple[int, int]]:
    room = []
    width_edge = right_corner[0] - left_corner[0]
    height_edge = right_corner[1] - left_corner[1]

    for col in range(width_edge+1):
        offset = left_corner[0] + col
        room.append((offset, left_corner[1])) # top edge
        room.append((offset, left_corner[1] + height_edge)) # bottom edge

    for row in range(height_edge):
        offset = left_corner[1] + row
        room.append((left_corner[0], offset)) # left edge
        room.append((left_corner[0] + width_edge, offset)) # right edge
    
    door_cor = room[5]
    room = room[0:5] + room[6:]
    return room, door_cor
