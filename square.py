#!/usr/bin/python3

import sys
import numpy as np
import numpy.linalg as la
from pyhull.halfspace import Halfspace, HalfspaceIntersection
from mpmath import mp, matrix

def unit(v):
  n = la.norm(v)
  return (1.0/n) * v

def ccw(v):
  return np.array([-v[1], v[0]])

def theta_t(theta):
  return 1.0 - np.tan(np.pi/4 - theta)

colorShades = np.array([[0.196, 0.533, 0.741],
                        [0.398, 0.684, 0.660],
                        [0.600, 0.835, 0.580],
                        [0.902, 0.961, 0.596],
                        [0.996, 0.878, 0.545],
                        [0.992, 0.715, 0.447],
                        [0.988, 0.553, 0.349],
                        [0.835, 0.243, 0.310],   # highest level
                        [0.576, 0.439, 0.859],   # for extended cover
                        [0.641, 0.390, 0.722],
                        [0.705, 0.341, 0.584],
                        [0.770, 0.292, 0.447],
                        [0.835, 0.243, 0.310]])  # again highest level

blueIndex = 9

# ------------------------------------------------------------------------------------------

class SquareRecorder():
  """Record areas of the square that have been checked, and save the final result."""

  def __init__(self, pieces):
    self.pieces = pieces
    self.square = np.zeros((pieces, pieces), np.int8)
    sys.stderr.write("Pieces = %d\n" % (self.pieces))

  def add(self, x0, x1, y0, y1, level, overlapOk=False):
    for x in range(x0, x1):
      for y in range(y0, y1):
        assert overlapOk or self.square[x, y] == 0, (x0, x1, y0, y1, x, y, level, self.square[x, y])
        self.square[x, y] = level

  def isDone(self, x0, x1, y0, y1):
    for x in range(x0, x1):
      for y in range(y0, y1):
        if self.square[x, y] == 0:
          return False
    return True
    
  def axis(self, out, width, theta, label):
    r = 105.0 + width
    r1 = r + 4
    r2 = r + 4
    y = 100.0 + width * theta_t(theta)
    out.write('<path>95 %g m %g %g l</path>\n' % (y, r, y))
    out.write('<path>%g 95 m %g %g l</path>\n' % (y, y, r))
    out.write('<text pos="%g %g" valign="center" style="math">%s</text>\n' % (r1, y, label))
    out.write('<text pos="%g %g" valign="baseline" halign="center" style="math">%s</text>\n' % (y, r2, label))
      
  def write(self, fname, width, axes=[]):
    r = 100 + width
    out = open(fname, "w")
    out.write('<?xml version="1.0"?>\n')
    out.write('<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
    out.write('<ipe version="70221" creator="square.py">\n')
    out.write('<page>\n')
    out.write('<path>100 100 m %g 100 l %g %g l 100 %g l h</path>\n' % (r, r, r, r))
    out.write('<image rect="100 100 %g %g" width="%d" height="%d" ColorSpace="DeviceRGB">\n'
              % (r, r, self.pieces, self.pieces))
    colors = ["ffffff"] # 0 -- not checked
    for k in range(colorShades.shape[0]):
      col = colorShades[k]
      colors.append("%02x%02x%02x" % (int(col[0] * 256), int(col[1] * 256), int(col[2] * 256)))
    for y in range(self.pieces-1, -1, -1):
      for x in range(self.pieces):
        out.write(colors[self.square[x, y]])
      out.write('\n')
    out.write('</image>\n')
    for t, label in axes:
      self.axis(out, width, t, label)
    out.write('<path arrow="normal">80 80 m %g 80 l</path>\n' % (130 + width))
    out.write('<path arrow="normal">80 80 m 80 %g l</path>\n' % (130 + width))
    out.write('<text pos="%g 86" size="12" style="math">\\theta\'</text>\n' % (130 + width))
    out.write('<text pos="70 %g" size="12" style="math">\\theta</text>\n' % (115 + width))
    out.write('</page>\n')
    out.write('</ipe>\n')
    out.close()

# ------------------------------------------------------------------------------------------

class MPCover():
  "Check that pentagon lies in the cover with arbitrary precision."
  def __init__(self, side, extended, distance):
    self.distance = mp.mpf(distance)
    rightPointX = mp.mpf('1.41949302') if extended else mp.sqrt(2)
    self.side = side                   # side length of square
    self.d = side/mp.cos(mp.pi/12)     # length of left edge
    sin60 = mp.sqrt(3) / 2
    self.top = matrix([self.d / 2, sin60 * self.d])
    self.right = matrix([side * rightPointX, 0])
    self.nleft = matrix([[-sin60, 0.5]])
    self.ndown = matrix([[0, -1]])
    v = self.right - self.top
    norm = mp.norm(v)
    self.nright = matrix([[-v[1] / norm, v[0] / norm]])
    self.offset = self.nright * self.right

  def verify(self, x0, x1, y0, y1, s, t, x, y):
    "Verify the transformation given by (s, t, x, y) on pentagon."
    pentagon = [(x0, 0), (x1, 0), (0, y0), (0, y1), (self.side, self.side)]
    # reference point in interior of pentagon
    ref = matrix([x0 + x1 + self.side, y0 + y1 + self.side]) / 5
    translation = matrix([s * ref[0] + t * ref[1] + x, -t * ref[0] + s * ref[1] + y])
    # reconstruct rotation angle
    norm = mp.sqrt(s*s + t*t)
    if t < 0:
      angle = -mp.acos(s / norm)
    else:
      angle = mp.acos(s / norm)
    rotation = matrix([[ mp.cos(angle), mp.sin(angle)],
                       [-mp.sin(angle), mp.cos(angle)]])
    pentagon = list(map(lambda pt: matrix(pt), pentagon))
    ipentagon = list(map(lambda pt: rotation * (pt - ref) + translation, pentagon))
    # Verify that ipentagon is a congruent image of pentagon
    eps = mp.mpf(10)**-(mp.dps - 2)  # accept error in last two digits
    for i in range(5):
      for j in range(i+1, 5):
        d1 = mp.norm(pentagon[j] - pentagon[i])
        d2 = mp.norm(ipentagon[j] - ipentagon[i])
        dd = abs(d1 - d2)
        assert dd < eps, (pentagon, dd, eps)
    dists = []
    for p in ipentagon:
      dists.append(self.nleft * p)
      dists.append(self.ndown * p)
      dists.append(self.nright * p - self.offset)
    dist = max(map(lambda m: m[0], dists))
    if dist > -self.distance:
      sys.stderr.write("Pentagon failing slack test: %d %d %d %d\n" % (x0, x1, y0, y1))
      return False
    return True

# ------------------------------------------------------------------------------------------

# 4-dimensional vectors (s, t, x, y)
# where s = lambda cos theta, t = lambda sin theta
# (s, t, x, y) represents the transformation mapping (a, b)
# to (s*a + t*b + x, -t*a + s*b + y)
# that is: rotate by theta around origin, scale by lambda, translate by (x,y)

class Cover():
  def __init__(self, side, extended, distance):
    # according to Maple, the value is L0 = 1.419493024
    rightPointX = 1.4194 if extended else np.sqrt(2.0)
    self.side = side                   # side length of square
    self.d = side/np.cos(np.pi/12.0)   # length of left edge
    sin60 = 0.5 * np.sqrt(3)
    self.top = np.array([0.5 * self.d, sin60 * self.d])
    self.right = np.array([side * rightPointX, 0])
    self.nleft = np.array([-sin60, 0.5])
    self.ndown = np.array([0.0, -1.0])
    v = self.right - self.top
    self.nright = ccw(unit(v))
    self.offset = -np.dot(self.nright, self.right)
    self.mp = MPCover(side, extended, distance)
    
  def __repr__(self):
    return "top=%s, right=%s" % (self.top, self.right)

  def constraint(self, p, normal, offset):
    "Return halfspace that constrains this point to lie inside the half-plane."
    # (s*a + t*b + x) * nx + (-t*a + s*b + y) * ny <= offset
    # (a * nx + b * ny) * s + (b * nx - a * ny) * t + nx * x + ny * y <= offset
    return Halfspace([np.dot(p, normal),
                      p[1] * normal[0] - p[0] * normal[1],
                      normal[0],
                      normal[1]], offset)

  def constraints(self, p):
    "Return list of three halfspaces that constrain this point to lie in the cover."
    return [self.constraint(p, self.ndown, 0.0),
            self.constraint(p, self.nleft, 0.0),
            self.constraint(p, self.nright, self.offset)]

  def check(self, x0, x1, y0, y1):
    points = np.array([[x0, 0], [x1, 0], [0, y0], [0, y1], [self.side, self.side]])
    halfspaces = []
    for pt in points:
      halfspaces += self.constraints(pt)
    hi = HalfspaceIntersection(halfspaces, [0.0, 0.0, self.top[0], self.top[1] / 2.0])
    # find best vertex
    bestNorm = 0
    bestV = None
    for v in hi.vertices:
      n = la.norm(v[:2])
      if n > bestNorm:
        bestNorm = n
        bestV = v
    if bestNorm < 1.0:
      return False
    return self.mp.verify(x0, x1, y0, y1, bestV[0], bestV[1], bestV[2], bestV[3])

# ------------------------------------------------------------------------------------------

def fillSquare(cover, r, x0, x1, y0, y1, level):
  if r.isDone(x0, x1, y0, y1):
    return
  if cover.check(x0, x1, y0, y1) or cover.check(y0, y1, x0, x1):
    r.add(x0, x1, y0, y1, level)
    if x0 != y0:
      r.add(y0, y1, x0, x1, level)
    elif x0 == 0 and level == blueIndex:
      r.add(y0, y1, x0, x1, level, True)
  elif x1 - x0 > 1 and y1 - y0 > 1:
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2
    fillSquare(cover, r, x0, xm, y0, ym, level+1)
    fillSquare(cover, r, x0, xm, ym, y1, level+1)
    fillSquare(cover, r, xm, x1, y0, ym, level+1)
    fillSquare(cover, r, xm, x1, ym, y1, level+1)

def fillAll():
  side = 160 # side length of square
  d = 32     # starting square size
  cover = Cover(side, False, '0.001')
  r = SquareRecorder(side)
  for x in range(side // d):
    for y in range(side // d):
      fillSquare(cover, r, d * x, d * x + d, d * y, d * y + d, 1)
  sys.stderr.write("Adding three regions using extended cover (no recursion)\n")
  ecover = Cover(side, True, '0.01')
  fillSquare(ecover, r, 0, 1, 0, side - 1, blueIndex)
  fillSquare(ecover, r, 1, 3, side - 1, side, blueIndex + 2)
  fillSquare(ecover, r, 0, 1, side - 1, side, blueIndex + 3)
  axes = [(np.pi/6, "\\pi/6"),
          (np.radians(35), "35^\\circ"),
          (np.radians(25), "25^\\circ"),
          (0.049, "\\theta_0"),
          (np.pi/16, "\\pi/16")]
  r.write("square.ipe", 200, axes)

# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
  mp.dps = 50  # digits of precision for high-precision check
  fillAll()

# ------------------------------------------------------------------------------------------
