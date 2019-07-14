def find_waypoints(
    frame, horizontal_stretch, statistic,
    max_angle=30*np.pi/180, num_angles=24, steps=1, step_len=20,
    sphere_radius=35, draw_waypoints=True, print_debug=False,
):
    # This is the center of the frame and where the car is
    starting_point = np.array([frame.shape[0], frame.shape[1] // 2])

    waypoints = np.zeros((steps+1, 2))
    waypoints[0] = starting_point
    angle = 1.5 * np.pi  # "Go straight" angle
    point = starting_point
    for step in range(steps):
        scores = []
        for angle_idx in range(-num_angles, num_angles+1):
            # We're iterating over candidate angles that for a cone of [-40, 40] degrees
            angle_i = angle + angle_idx * max_angle / num_angles
            vector = step_len * np.array([np.sin(angle_i), horizontal_stretch*np.cos(angle_i)])
            new_position = point + vector

            lower_bound, upper_bound = (
                (new_position - sphere_radius).astype(int),
                (new_position + sphere_radius).astype(int),
            )
            x, y = np.meshgrid(
                np.arange(max(0, lower_bound[0]), min(frame.shape[0], upper_bound[0])),
                np.arange(max(0, lower_bound[1]), min(frame.shape[1], upper_bound[1])),
            )
            xy = np.array([x.flatten(), y.flatten()]).T

            which_in_sphere = (np.linalg.norm(xy - new_position, axis=1) < sphere_radius)

            score = statistic(frame[xy[:, 0], xy[:, 1]])
            scores.append([score, -abs(angle_idx), angle_i, new_position])

            if draw_waypoints:
                x_idx, y_idx = np.round(new_position).astype(int)
                frame[x_idx-1:x_idx+1, y_idx-1:y_idx+1] = score

        # We choose the waypoint that has the best score, but if two waypoints
        #  have the same score, we choose the one that's closest to going the
        #  same direction as previously
        best_score, best_angle_idx, best_angle, best_new_position = max(
            scores, key=itemgetter(0, 1)
        )
        point = best_new_position
        angle = best_angle
        waypoints[step+1] = point

        if draw_waypoints:
            x_idx, y_idx = np.round(best_new_position).astype(int)
            # FIXME: sometimes a waypoint goes around the frame
            frame[x_idx-1:x_idx+1, y_idx-1:y_idx+1] += 1

        if print_debug:
            print('score\t\tangle_i')
            for score, _1, angle_i, _2 in scores:
                print('{:.6f}\t\t{}'.format(score, angle_i))

    return waypoints, frame


def find_angle(waypoints, starting_waypoint=0):
    difference_quotient = (waypoints[-1, 1] - waypoints[starting_waypoint, 1]) / (waypoints[-1, 0] - waypoints[starting_waypoint, 0])
    return -np.arctan(difference_quotient)
