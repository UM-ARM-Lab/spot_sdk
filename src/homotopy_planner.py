from time import sleep
import cv2

import matplotlib.pyplot as plt
import numpy as np


def f0(z):
    # assume no more than 3 obstacles, otherwise the polynomial should be
    # higher order. I think?
    return z ** 3 + 2 * z ** 2 + 3 * z + 4


def F(z, zetas):
    """
    Args:
        z: complex number, i.e an (x, y) point
        zetas: list of complex numbers representing the obstacles (centroids)
    """
    return f0(z) / np.prod([z - zeta for zeta in zetas])


def integrate(tau, zetas):
    """
    Integrate F(z)dz from the start point to the goal point
    Args:
        tau: a function that can be evaluated from 0 to 1 and returns the path position
        zetas: list of complex numbers representing the obstacles (centroids)
    """
    # numerically integrate F(tau(t))tau'(t)dt from 0 to 1
    integral = 0
    dt = 0.01
    for t in np.arange(0, 1, dt):
        dz = tau(t + dt) - tau(t)
        z = tau(t)
        integral += F(z, zetas) * dz
    return integral


def make_tau(start, goal, waypoint):
    def _tau(t):
        """
        Piecewise linear path: start --> waypoint --> goal.
        The "time" is arbitrary, so we place 0.5 at the waypoint.
        """
        if t <= 0.5:
            return start + 2 * t * (waypoint - start)
        else:
            return waypoint + 2 * (t - 0.5) * (goal - waypoint)

    return _tau


def compare_waypoints(fig, ax, start, goal, waypoint1, waypoint2, zetas):
    tau1 = make_tau(start, goal, waypoint1)
    tau2 = make_tau(start, goal, waypoint2)
    windings = []
    for zeta in zetas:
        # to compute the winding number for a given obstacle,
        # go from 0 to 1 and compute the angle between
        # the vector from the obstacle to the path and the +x axis
        # then integrate these angles and you'll get a total rotation of either 0 or 2pi???
        dt = 0.01
        integral_angle = 0
        last_angle = None
        for t in np.arange(0, 2, dt):
            if t < 1:
                z = tau1(t)
            else:
                z = tau2(2 - t)
            ax.plot([zeta[0], z[0]], [zeta[1], z[1]], c='y', linewidth=0.5)
            ax.scatter(z[0], z[1], c='k', s=1)
            angle = np.arctan2(z[1] - zeta[1], z[0] - zeta[0])
            if last_angle is not None:
                integral_angle += (angle - last_angle)
            last_angle = angle
        fig.show()

        # round to nearest multiple of 2 pi
        winding_number = np.round(integral_angle / (2 * np.pi)) * 2 * np.pi
        windings.append(winding_number)
    return np.array(windings)


def viz_tau(ax, tau, c):
    xs = []
    ys = []
    for t in np.linspace(0, 1, 100):
        z = tau(t)
        xs.append(z[0])
        ys.append(z[1])
    ax.plot(xs, ys, c=c)


def main():
    cv2.waitKey(1)

    obstacle_poly = np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ])

    zetas = np.array([
        [0., 0.],
    ])
    start = np.array([-2, 0])
    goal = np.array([2, 0])

    fig, ax = plt.subplots()
    ax.fill(obstacle_poly[:, 0], obstacle_poly[:, 1], c='k')
    ax.axis("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    same_homo = compare_waypoints(fig, ax, start, goal, np.array([0, 1.4]), np.array([0, 1.6]), zetas)

    fig, ax = plt.subplots()
    ax.fill(obstacle_poly[:, 0], obstacle_poly[:, 1], c='k')
    ax.axis("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    diff_homo = compare_waypoints(fig, ax, start, goal, np.array([0, 1.4]), np.array([0, -1.6]), zetas)

    print(same_homo, diff_homo)


if __name__ == '__main__':
    main()
