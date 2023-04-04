# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:02:21 2021

@author: pcjcl
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from datetime import datetime
import itertools


# %%
def calculate_total_edge_length(coordinates):
    length = 0
    for p in range(1, len(coordinates)):
        # add up the differences between all the points
        length += ((coordinates[p][0] - coordinates[p - 1][0]) ** 2 + (
                coordinates[p][1] - coordinates[p - 1][1]) ** 2) ** 0.5

    return length


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def curved_lines(x_vals,
                 y_vals):
    # make curve lines between all points even if they go back and forth on grid rather than in order
    x_curve = []
    y_curve = []
    for i in range(0, len(x_vals) - 1):

        # solution to quadratic equation
        b = (x_vals[i + 1] - x_vals[i]) / 2 + x_vals[i]
        a = -1
        c = (b - x_vals[i]) ** 2

        if y_vals[i + 1] <= y_vals[i] and x_vals[i + 1] > x_vals[i]:
            a = 1
            c = -(b - x_vals[i]) ** 2
        elif y_vals[i + 1] > y_vals[i] and x_vals[i + 1] <= x_vals[i]:
            a = 1
            c = -(b - x_vals[i]) ** 2

        # normalise bulge (height of quadratic part) by the length of the line
        length = ((y_vals[i + 1] - y_vals[i]) ** 2 + (x_vals[i + 1] - x_vals[i]) ** 2) ** 0.5
        norm = length ** 0.3 / (abs(c) * 20)
        if y_vals[i + 1] < y_vals[i]:
            norm = -length ** 0.3 / (abs(c) * 20)

        m = (y_vals[i + 1] - y_vals[i]) / (x_vals[i + 1] - x_vals[i])  # linear part y = mx * d
        d = y_vals[i] - m * x_vals[i]
        x_ = np.linspace(x_vals[i], x_vals[i + 1], 50)
        y_ = m * x_ + d + (a * (x_ - b) ** 2 + c) * norm

        x_curve.append(np.copy(x_))
        y_curve.append(np.copy(y_))
    x_curve = list(itertools.chain(*x_curve))
    y_curve = list(itertools.chain(*y_curve))

    return x_curve, y_curve


def generate_path_conditions(options=''):
    # physical constants
    min_cloud_height = 2000  # bottom of cloud height, m above sesa level
    max_cloud_height = 5000  # starting height of cloud top of snowflake, m
    height_increment = 50  # change in height for updating snowflake, m
    temperature_at_cloud_top = -40  # top of cloud T, degrees C
    temperature_at_cloud_bottom = -6  # bottom of cloud T, degrees C
    ground_temperature = -5 * np.random.default_rng().random()  # ground T randomly generated between 0 to -5
    m_p = 1.67262191E-27  # mass of proton, kg
    m_h2o = 18 * m_p  # mass of water molecule
    k = 1.38064852E-23  # Boltzman constant

    if options != '':
        start_height = int(options['height'])
        start_relative_humidity = int(options['humidity'])
    else:
        start_height = 2500 + 2500 * np.random.default_rng().random()
        start_relative_humidity = 113

    height_path = np.linspace(max_cloud_height, 0, num=round(max_cloud_height / height_increment) + 1)
    start_idx = find_nearest(height_path, start_height)

    temperature_path = generate_temperature_path(ground_temperature, height_increment, max_cloud_height,
                                                 min_cloud_height, temperature_at_cloud_bottom,
                                                 temperature_at_cloud_top)

    over100RHu, saturation_density_H2O_path = generate_water_saturation_density_path(height_increment, height_path, k,
                                                                                     m_h2o, max_cloud_height,
                                                                                     min_cloud_height,
                                                                                     start_relative_humidity,
                                                                                     temperature_path)

    # transform the temperature saturated water density path into growth_record array
    # form is [[growth_type, growth_rate]], TODO refactor into object
    growth_record = transform_temperature_and_water_density_to_growth_record(saturation_density_H2O_path, start_idx,
                                                                             temperature_path)

    if options != '':
        return growth_record, temperature_path, over100RHu, start_idx, height_path
    else:
        return growth_record


def transform_temperature_and_water_density_to_growth_record(saturation_density_H2O_path, start_idx, temperature_path):
    growth_record = [0, 0]
    for i in range(start_idx, len(temperature_path)):
        # the line from this graph
        # https://physics.montana.edu/demonstrations/video/1_mechanics/demos/snowflakegraph.html
        # formula from fitting 2nd order polynomial to the snowflake shape graph: -0.0005x2 - 0.017x
        snowflake_shape_line = -0.0004 * temperature_path[i] ** 2 - 0.0164 * temperature_path[i] + 0.0051

        if saturation_density_H2O_path[i] < 0:
            growth_type_probability_weights = [0, 0, 0, 1]

        elif saturation_density_H2O_path[i] <= snowflake_shape_line:
            rem = abs(snowflake_shape_line - saturation_density_H2O_path[i] / (snowflake_shape_line))
            growth_type_probability_weights = [rem, rem / 2, rem ** 3, 0]

        elif saturation_density_H2O_path[i] > snowflake_shape_line:
            rem = abs(saturation_density_H2O_path[i] - (snowflake_shape_line) / (saturation_density_H2O_path[i]))
            growth_type_probability_weights = [1 - rem / 2, 1 - rem, rem,
                                               0]

        growth_type = random.choices(['normal', 'stretch', 'branch', 'none'], weights=growth_type_probability_weights,
                                     k=1)

        # There is also a random element to growth.
        # Colder temperatures snow growth a bit higher, and also linear on density of water vapor
        growth_rate = saturation_density_H2O_path[i] * abs(temperature_path[
                                                               i]) ** 0.5 * np.random.default_rng().random()
        growth_record = np.vstack([growth_record, [growth_type, growth_rate]])
    growth_record = np.delete(growth_record, obj=0, axis=0)  # deletes the initial row of 0s.
    return growth_record


def generate_water_saturation_density_path(height_increment, height_path, k, m_h2o, max_cloud_height, min_cloud_height,
                                           start_relative_humidity, temperature_path):
    vapour_pressure_H2O = (
            0.0052 * temperature_path ** 3 + 0.7445 * temperature_path ** 2 + 35.976 * temperature_path + 598.87)
    sine_wave_period = (round((max_cloud_height - min_cloud_height) / height_increment)) \
                       * np.random.default_rng().random()
    relative_humidity_cloud_path = np.linspace(0, (round((max_cloud_height - min_cloud_height) / height_increment)),
                                               num=(round((max_cloud_height - min_cloud_height) / height_increment)))
    relative_humidity_cloud = start_relative_humidity + 4 * np.sin(relative_humidity_cloud_path / sine_wave_period)
    # add fluctuations to the relative humidity path
    relative_humidity_noise = np.random.normal(0, 1.2, round((max_cloud_height - min_cloud_height) / height_increment))
    relative_humidity_cloud = relative_humidity_cloud + relative_humidity_noise
    # random ground relative humidity between 0 and 100 %
    ground_relative_humidity = np.random.default_rng().random() * 100
    RHu_noise = np.random.normal(0, 15, len(height_path) - len(relative_humidity_cloud))
    relative_humidity_cloud = np.append(relative_humidity_cloud,
                                        np.linspace(relative_humidity_cloud[-1], ground_relative_humidity,
                                                    num=(len(height_path) - len(relative_humidity_cloud))))
    # adding noise with a std of 10% relative humidity
    relative_humidity_cloud[round((max_cloud_height - min_cloud_height) / height_increment):] \
        += np.random.normal(0, 5, len(height_path) - len(relative_humidity_noise))
    # stop RH going below 0 (not physically possible)
    for r in range(0, len(relative_humidity_cloud)):
        if relative_humidity_cloud[r] < 0:
            relative_humidity_cloud[r] = 0
    over100RHu = relative_humidity_cloud - 100 * np.ones(len(relative_humidity_cloud))
    p_H2O = relative_humidity_cloud * vapour_pressure_H2O / 100  # in Pa
    saturation_pressure_H2O = over100RHu / 100 * vapour_pressure_H2O
    # T in K remember, * 1000 makes it into grams
    saturation_density_H2O_path = m_h2o * saturation_pressure_H2O / (k * (temperature_path + 273.15)) * 1000
    return over100RHu, saturation_density_H2O_path


def generate_temperature_path(ground_temperature, height_increment, max_cloud_height, min_cloud_height,
                              temperature_at_cloud_bottom, temperature_at_cloud_top):
    temperature_path = np.linspace(temperature_at_cloud_top, temperature_at_cloud_bottom,
                                   num=round((max_cloud_height - min_cloud_height) / height_increment))
    temperature_path = np.append(temperature_path, np.linspace(temperature_at_cloud_bottom, ground_temperature,
                                                               num=round((min_cloud_height / height_increment)) + 1))
    # add fluctuation to temperature (noise)
    temperature_noise = np.random.normal(0, 1, len(temperature_path))
    temperature_path = temperature_path + temperature_noise
    return temperature_path


def grow_snowflakes_along_path(growth_record, options=''):
    branch_options = 0  # default
    if options != '':
        branch_options = options['branchopt']

    initial_radius_length = 1 / 2

    corner1 = np.array([initial_radius_length, 0])
    centre1a = np.array([0.75 * initial_radius_length, 3 ** 0.5 / 4 * initial_radius_length])

    flake_points_coordinates_np = np.array([0.1 * centre1a, 0.1 * corner1])
    flake_points_coordinates_tuple = tuple(map(tuple, flake_points_coordinates_np))

    # rotation matrices for branching at +- 60 degrees
    Rotate_plus60 = np.array(((np.cos(np.radians(60)), -np.sin(np.radians(60))),
                              (np.sin(np.radians(60)), np.cos(np.radians(60)))))
    Rotate_minus60 = np.array(((np.cos(np.radians(-60)), -np.sin(np.radians(-60))),
                               (np.sin(np.radians(-60)), np.cos(np.radians(-60)))))
    # reflection in x-axis matrix
    Reflect_in_x = np.array(((1, 0), (0, -1)))

    branch_direction1_vector = np.matmul(corner1 / np.linalg.norm(corner1), Rotate_minus60)
    branch_direction2_vector = np.matmul(corner1 / np.linalg.norm(corner1), Rotate_plus60)
    branch_vectors = np.array((1, 1))
    branch_origin = np.array((1, 1))

    all_flakes = [np.copy(flake_points_coordinates_np)]  # first one is the hexagon

    number_of_branches = 0

    for idx in range(0, len(growth_record[:, ])):
        edge_length = (2 * calculate_total_edge_length(flake_points_coordinates_tuple)) ** 0.5
        area = growth_record[idx, 1] / 6  # /6 because it's doing it on 6 sides
        if growth_record[idx, 0] == ['none']:
            pass
        if growth_record[idx, 0] == ['normal']:
            for point in flake_points_coordinates_np:
                point = grow_centre_and_corner_points(area, centre1a, corner1, edge_length, number_of_branches, point)

                if number_of_branches > 0:  # add branch points growth
                    for branch_number in range(0, len(branch_vectors)):
                        grow_existing_branches(area, branch_number, branch_options, branch_origin, branch_vectors,
                                               edge_length, number_of_branches, point)

            # this part grows the branch origins outwards too
            if number_of_branches > 0:
                grow_origins_of_branches(area, edge_length, edget, flake_points_coordinates_np)

        if growth_record[idx, 0] == ['stretch']:  # just make it so that only outer corners grow
            for point in flake_points_coordinates_np:
                if np.allclose(point / np.linalg.norm(point), corner1 / np.linalg.norm(corner1), rtol=1e-05) == True:
                    point += 0.5 * area * corner1  # stretch the corner points

        if growth_record[idx, 0] == ['branch']:
            branch_origin, branch_vectors, edget, flake_points_coordinates_np, number_of_branches = create_new_branch(
                Reflect_in_x, area, branch_direction1_vector, branch_direction2_vector, branch_origin, branch_vectors,
                corner1,
                edge_length, flake_points_coordinates_np, number_of_branches)

        flake_points_coordinates_tuple = tuple(map(tuple, flake_points_coordinates_np))  # overwrite points tuple
        all_flakes.append(np.copy(flake_points_coordinates_np))

    all_flakes_with_rotation = rotate_6_times_to_finish_snowflakes(all_flakes)

    return all_flakes_with_rotation, corner1, centre1a


def rotate_6_times_to_finish_snowflakes(all_flakes):
    all_flakes_with_rotation = []
    for f in all_flakes:
        temporary_rotated_flake = f
        for n in range(1, 6):
            theta = np.radians(60 * n)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            for point in f:
                temporary_rotated_flake = np.vstack([temporary_rotated_flake, [np.matmul(point, R)]])

        all_flakes_with_rotation.append(np.copy(temporary_rotated_flake))
    return all_flakes_with_rotation


def create_new_branch(Reflect_in_x, area, branch_direction1_vector, branch_direction2_vector, branch_origin,
                      branch_vectors,
                      corner1, edge_length, flake_points_coordinates_np, number_of_branches):
    for point in range(0, len(flake_points_coordinates_np)):
        # branching off centre line only. can add elifs for branches branching from branches later
        if np.allclose(flake_points_coordinates_np[point] / np.linalg.norm(flake_points_coordinates_np[point]),
                       corner1 / np.linalg.norm(corner1),
                       rtol=1e-05):
            # normalise branch width by the value (length) of e
            # (stops the branches getting out of control chunk as the snowflake grows)
            branch_width = 1 / flake_points_coordinates_np[point][
                0]
            if branch_width > 0.15:
                branch_width = 0.14
            branch_origin = np.vstack([branch_origin, [
                (0.85 + branch_width / 2) * (flake_points_coordinates_np[point] - flake_points_coordinates_np[0]) +
                flake_points_coordinates_np[0]],
                                       [np.matmul((0.85 + branch_width / 2) * (
                                               flake_points_coordinates_np[point] - flake_points_coordinates_np[
                                           0]) + flake_points_coordinates_np[0],
                                                  Reflect_in_x)]])

            temporary_points_coordinates = np.copy(flake_points_coordinates_np)
            if number_of_branches < 1:
                # remove the original corner for now
                # so it can be put in the middle of the branches axis = 0 makes it keep its shape
                temporary_points_coordinates = np.delete(temporary_points_coordinates, point, axis=0)
                temporary_points_coordinates = np.vstack([temporary_points_coordinates, [
                    0.85 * (flake_points_coordinates_np[point] - flake_points_coordinates_np[0]) +
                    flake_points_coordinates_np[0]],
                                                          [((0.85 + branch_width / 2) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                            flake_points_coordinates_np[
                                                                0] + branch_direction1_vector * area / edge_length)],
                                                          [(0.85 + branch_width) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                           flake_points_coordinates_np[0]],
                                                          [flake_points_coordinates_np[point]],
                                                          [np.matmul((0.85 + branch_width) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                                     flake_points_coordinates_np[0], Reflect_in_x)],
                                                          [(np.matmul((0.85 + branch_width / 2) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                                      flake_points_coordinates_np[0],
                                                                      Reflect_in_x) + branch_direction2_vector * area / edge_length)],
                                                          [np.matmul(
                                                              0.85 * (flake_points_coordinates_np[point] -
                                                                      flake_points_coordinates_np[0]) +
                                                              flake_points_coordinates_np[0], Reflect_in_x)]])
                flake_points_coordinates_np = np.copy(temporary_points_coordinates)

            elif number_of_branches >= 1:
                # delete outer point corner point on x-axis and everything after it
                temporary_points_coordinates = np.delete(temporary_points_coordinates,
                                                         np.s_[point:len(flake_points_coordinates_np)],
                                                         axis=0)

                # put in the new branch coordinates as before with the outer point in the centre
                temporary_points_coordinates = np.vstack([temporary_points_coordinates, [
                    0.85 * (flake_points_coordinates_np[point] - flake_points_coordinates_np[0]) +
                    flake_points_coordinates_np[0]],
                                                          [((0.85 + branch_width / 2) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                            flake_points_coordinates_np[
                                                                0] + branch_direction1_vector * area / edge_length)],
                                                          [(0.85 + branch_width) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                           flake_points_coordinates_np[0]],
                                                          [flake_points_coordinates_np[point]],
                                                          [np.matmul((0.85 + branch_width) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                                     flake_points_coordinates_np[0], Reflect_in_x)],
                                                          [(np.matmul((0.85 + branch_width / 2) * (
                                                                  flake_points_coordinates_np[point] -
                                                                  flake_points_coordinates_np[0]) +
                                                                      flake_points_coordinates_np[0],
                                                                      Reflect_in_x) + branch_direction2_vector *
                                                            area / edge_length)],
                                                          [np.matmul(
                                                              0.85 * (flake_points_coordinates_np[point] -
                                                                      flake_points_coordinates_np[0]) +
                                                              flake_points_coordinates_np[0], Reflect_in_x)]])
                # put the rest of the points you deleted back in
                for k in range(point + 1, len(flake_points_coordinates_np)):
                    temporary_points_coordinates = np.vstack(
                        [temporary_points_coordinates, [flake_points_coordinates_np[k]]])
                flake_points_coordinates_np = np.copy(temporary_points_coordinates)
            branch_vectors = np.vstack(
                [branch_vectors, [branch_direction1_vector], [branch_direction2_vector]])  # add the growth vectors
            break
    number_of_branches += 1
    return branch_origin, branch_vectors, temporary_points_coordinates, flake_points_coordinates_np, number_of_branches


def grow_existing_branches(area, branch_number, branch_options, branch_origin, branch_vectors, edge_length,
                           number_of_branches, point):
    # update the growth rate only on the even numbers so that the +- pairs grow equally
    branch_growth = number_of_branches * area / edge_length
    # finds the points in the branching directions
    if np.allclose((point - branch_origin[branch_number]) / np.linalg.norm(point - branch_origin[branch_number]),
                   branch_vectors[branch_number] / np.linalg.norm(branch_vectors[branch_number]), rtol=1e-05):
        # advanced mode branches overlapping option is when branchopt == 1
        if branch_options != 1 and abs(point[1] + branch_growth * branch_vectors[branch_number][1]) > (
                point[0] + branch_growth ** branch_vectors[branch_number][
            0]) * 0.42:
            # stop the branches from crossing over.
            # (I thought it should be < x*0.577 / tan30 but seems to still cross)
            pass
        else:
            # grow the branches without restriction
            point += branch_growth * branch_vectors[branch_number]


def grow_origins_of_branches(area, edge_length, edget, flake_points_coordinates_np):
    for i in range(0, len(flake_points_coordinates_np)):
        if np.allclose(flake_points_coordinates_np[i], edget[i], rtol=1e-07):
            if flake_points_coordinates_np[i][1] > 0:
                flake_points_coordinates_np[i] += 1.2 * area / edge_length * (np.array((0, 1)))

            elif flake_points_coordinates_np[i][1] < 0:
                flake_points_coordinates_np[i] += 1.2 * area / edge_length * (np.array((0, -1)))


def grow_centre_and_corner_points(area, centre1a, corner1, edge_length, number_of_branches, point):
    # checks for points with the same direction vector as the original CORNER point
    # (from origin) then grows it
    normalised_e_vector = point / np.linalg.norm(point)
    # allclose works better than array_equal because has tolerance in case of stupid float maths
    if np.allclose(normalised_e_vector, corner1 / np.linalg.norm(corner1),
                   rtol=1e-05):
        point += 10.1 * area / edge_length * corner1
    # checks for points (e) with the same direction vector as the original CENTRE point (from origin) then grows it
    elif np.allclose(normalised_e_vector, centre1a / np.linalg.norm(centre1a), rtol=1e-05):
        if number_of_branches < 1:
            point += 1 * area / edge_length * centre1a
        else:
            # slow down centre faces of starting hexagon growth after first branch
            point += 0.1 * area / edge_length * centre1a
    return point


def snowflake_animation(g, all_flakes_rot, snowflake):
    x_values = []
    y_values = []
    for s in all_flakes_rot[g]:
        x_values = np.append(x_values, s[0])
        y_values = np.append(y_values, s[1])
    x_values = np.append(x_values, all_flakes_rot[g][0][0])  # complete the pattern
    y_values = np.append(y_values, all_flakes_rot[g][0][1])
    x_c, y_c = curved_lines(x_values, y_values)
    snowflake.set_xdata(x_c)
    snowflake.set_ydata(y_c)

    return snowflake


def flake_video(all_flakes_rot, corner1, centre1a, options=''):
    x_values = []
    y_values = []
    Rottemp = np.array([centre1a, corner1])
    for n in range(1, 6):
        theta = np.radians(60 * n)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
    for e in range(0, len(all_flakes_rot[0])):
        x_values = np.append(x_values, all_flakes_rot[0][e][0])
        y_values = np.append(y_values, all_flakes_rot[0][e][1])
    x_values = np.append(x_values, all_flakes_rot[0][0][0])  # complete the pattern
    y_values = np.append(y_values, all_flakes_rot[0][0][1])
    x_c, y_c = curved_lines(x_values, y_values)

    fig, ax = plt.subplots()

    ax.set_xlim(-1.2 * np.max(all_flakes_rot[-1]), 1.2 * np.max(all_flakes_rot[-1]))
    ax.set_ylim(-1.2 * np.max(all_flakes_rot[-1]), 1.2 * np.max(all_flakes_rot[-1]))
    if options != '':
        sizeopt = options['sizeopt']
        if sizeopt == 1:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
    ax.set_facecolor('k')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    snowflake, = ax.plot(x_c, y_c, linewidth=1)
    # snowflake, = ax.fill_between(x_values, y_values,color=[(0.8,0.9,1)])

    flakeanimation = animation.FuncAnimation(fig, func=snowflake_animation,
                                             frames=np.arange(0, len(all_flakes_rot), 1),
                                             interval=100, fargs=(
            all_flakes_rot, snowflake,))  # interval in ms, frames is the i values
    # plt.show
    plt.close()

    # to hard save file:
    dt_now = str(datetime.now())
    dt_now = dt_now[:19]  # current time up to the seconds
    dt = dt_now.replace(':', '-')  # formatting so it can be used as a file name
    animationfilename = 'snowflakeanimation_' + dt + '.gif'

    print('abouttosave')
    flakeanimation.save((animationfilename), writer=animation.PillowWriter(
        fps=10))  # could not work out how to save animation.funcanimation object in the binary so saving real file for now and deleting after sending on server
    print('flake animation successfully created')
    return animationfilename


def othergraphs_animation(g, xdata, y1data, y2data, h, T, over100RHu, line1, line2):
    xdata.append(h[g])
    y1data.append(T[g])
    y2data.append(over100RHu[g] + 100)
    line1.set_xdata(xdata)
    line1.set_ydata(y1data)
    line2.set_xdata(xdata)
    line2.set_ydata(y2data)

    return line1, line2


def conditions_video(h, T, over100RHu, startidx):
    xdata, y1data, y2data = [], [], []

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Altitude (m)', fontsize=20)
    ax1.set_ylabel('Temperature ($^\circ$ C)', color='r', fontsize=20)
    ax2 = ax1.twinx()  #
    ax2.set_ylabel('Relative Humidity %', color='b', fontsize=20)

    ax1.set_xlim(0, h[startidx])
    ax1.set_ylim(min(T), max(T))
    ax1.invert_xaxis()
    ax2.set_ylim(min(over100RHu) + 100, max(over100RHu) + 100)
    line1, = ax1.plot(h[startidx], T[startidx], linestyle='', marker='x', color='r')
    line2, = ax2.plot(h[startidx], T[startidx], linestyle='', marker='x', color='b')

    othergraphsanimation = animation.FuncAnimation(fig, func=othergraphs_animation,
                                                   frames=np.arange(startidx, len(T), 1), interval=100,
                                                   repeat=True, fargs=(
            xdata, y1data, y2data, h, T, over100RHu, line1, line2))  # interval in ms, frames is the i values

    # to hard save file:
    dt_now = str(datetime.now())
    dt_now = dt_now[:19]  # current time up to the seconds
    dt = dt_now.replace(':', '-')  # formatting so it can be used as a file name
    extragraphsfilename = 'extraanimation_' + dt + '.gif'

    print('abouttosaveextragraphs')
    othergraphsanimation.save((extragraphsfilename), writer=animation.PillowWriter(
        fps=10))  # could not work out how to save animation.funcanimation object in the binary so saving real file for now and deleting after sending on server
    print('extragraphs animation successfully created')
    return extragraphsfilename
