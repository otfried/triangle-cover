#!/usr/bin/python3

import numpy as np
import numpy.linalg as la
from pyhull.halfspace import Halfspace, HalfspaceIntersection

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
    self.delta = 1.0 / pieces
    self.square = np.zeros((pieces, pieces), np.int8)
    print("Pieces = %d, delta = %g" % (self.pieces, self.delta))

  def t0_to_index(self, t):
    return int((t + 0.5 * self.delta) / self.delta)

  def t1_to_index(self, t):
    return int((t - 0.5 * self.delta) / self.delta)
    
  def add(self, x0, x1, y0, y1, level, overlapOk=False):
    xi0 = self.t0_to_index(x0)
    xi1 = self.t1_to_index(x1)    
    yi0 = self.t0_to_index(y0)
    yi1 = self.t1_to_index(y1)
    for x in range(xi0, xi1+1):
      for y in range(yi0, yi1+1):
        assert overlapOk or self.square[x, y] == 0, (x0, x1, y0, y1, x, y, level, self.square[x, y])
        self.square[x, y] = level

  def isDone(self, x0, x1, y0, y1):
    xi0 = self.t0_to_index(x0)
    xi1 = self.t1_to_index(x1)    
    yi0 = self.t0_to_index(y0)
    yi1 = self.t1_to_index(y1)
    for x in range(xi0, xi1+1):
      for y in range(yi0, yi1+1):
        if self.square[x, y] == 0:
          return False
    return True
    
  def dump(self):
    for y in range(self.pieces):
      for x in range(self.pieces):
        if self.square[x, y]:
          print("X", end="")
        else:
          print("-", end="")
      print()

  def axis(self, out, theta):
    y = 100.0 + 400.0 * theta_t(theta)
    out.write('<path>100 %g m 500 %g l</path>\n' % (y, y))
    out.write('<path>%g 100 m %g 500 l</path>\n' % (y, y))
      
  def write(self, fname):
    out = open(fname, "w")
    out.write('<?xml version="1.0"?>\n')
    out.write('<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
    out.write('<ipe version="70221" creator="square.py">\n')
    out.write('<page>\n')
    out.write('<path>100 100 m 500 100 l 500 500 l 100 500 l h</path>\n')
    out.write('<image rect="100 100 500 500" width="%d" height="%d" ColorSpace="DeviceRGB">\n'
              % (self.pieces, self.pieces))
    colors = ["ffffff"] # 0 -- not checked
    for k in range(colorShades.shape[0]):
      col = colorShades[k]
      colors.append("%02x%02x%02x" % (int(col[0] * 256), int(col[1] * 256), int(col[2] * 256)))
    for y in range(self.pieces-1, -1, -1):
      for x in range(self.pieces):
        out.write(colors[self.square[x, y]])
      out.write('\n')
    out.write('</image>\n')
    self.axis(out, np.pi/6)
    self.axis(out, np.radians(35))
    self.axis(out, np.radians(25))
    self.axis(out, 0.049)
    self.axis(out, np.pi/16)
    out.write('</page>\n')
    out.write('</ipe>\n')
    out.close()

# ------------------------------------------------------------------------------------------

# 4-dimensional vectors (s, t, x, y)
# where s = lambda cos theta, t = lambda sin theta
# (s, t, x, y) represents the transformation mapping (a, b)
# to (s*a + t*b + x, -t*a + s*b + y)
# that is: rotate by theta around origin, scale by lambda, translate by (x,y)

class Cover():
  def __init__(self, rightPointX, safety):
    self.d = 1.0/np.cos(np.pi/12.0)
    self.safety = safety
    sin60 = 0.5 * np.sqrt(3)
    self.top = np.array([0.5 * self.d, sin60 * self.d])
    self.right = np.array([rightPointX, 0.0])
    self.nleft = np.array([-sin60, 0.5])
    self.ndown = np.array([0.0, -1.0])
    v = self.right - self.top
    self.nright = ccw(unit(v))
    self.offset = -np.dot(self.nright, self.right)
    #print(self.nleft, self.nright, self.offset)
    #print(np.dot(self.nleft, self.top), np.dot(self.nright, self.top))
    
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
    points = np.array([[x0, 0], [x1, 0], [0, y0], [0, y1], [1, 1]])
    halfspaces = []
    for pt in points:
      halfspaces += self.constraints(pt)
    hi = HalfspaceIntersection(halfspaces, [0.0, 0.0, 0.5, 0.1])
    for v in hi.vertices:
      n = la.norm(v[:2])
      if n >= self.safety:
        return True
    return False

# ------------------------------------------------------------------------------------------

def fillSquare(cover, r, x0, x1, y0, y1, level, maxLevel):
  if r.isDone(x0, x1, y0, y1):
    return
  if cover.check(x0, x1, y0, y1) or cover.check(y0, y1, x0, x1):
    r.add(x0, x1, y0, y1, level)
    if x0 != y0:
      r.add(y0, y1, x0, x1, level)
    elif x0 == 0 and level == blueIndex:
      r.add(y0, y1, x0, x1, level, True)
  elif level < maxLevel:
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    fillSquare(cover, r, x0, xm, y0, ym, level+1, maxLevel)
    fillSquare(cover, r, x0, xm, ym, y1, level+1, maxLevel)
    fillSquare(cover, r, xm, x1, y0, ym, level+1, maxLevel)
    fillSquare(cover, r, xm, x1, ym, y1, level+1, maxLevel)

def fillAll():
  nsteps = 5
  maxLevel = 6
  cover = Cover(np.sqrt(2.0), 1.0001)
  ecover = Cover(1.4184, 1.0001)
  d = 1.0 / nsteps
  dd = d / 32.0  # for extended cover
  ddd = dd / 2.0
  r = SquareRecorder(int(1.0 / ddd))
  for x in range(nsteps):
    for y in range(nsteps):
      fillSquare(cover, r, d * x, d * x + d, d * y, d * y + d, 1, maxLevel)
  # add extended cover
  fillSquare(ecover, r, 0.0, dd, 0.0, 1.0 - dd, blueIndex, blueIndex) # do not recurse
  for x in range(4):
    fillSquare(ecover, r, x * dd, x * dd + dd, 1.0 - dd, 1.0, blueIndex + 2, blueIndex + 3)
  r.write("square.ipe")

# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
  fillAll()

# ------------------------------------------------------------------------------------------
