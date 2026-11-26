"""
Blender script: 3D schematic of a bent flexible device
(bronze substrate + polyimide + two gold electrodes with a nano-gap,
two holding rods on top at the ends, one pushing rod underneath).

HOW TO USE
  1. Open Blender (tested logic for 3.6 / 4.x / 5.x).
  2. Go to the "Scripting" workspace, click "New", paste this file, press "Run Script".
     (Or: blender --python bending_schematic_blender.py)
  3. The current scene is CLEARED and rebuilt. Change the parameters below and re-run
     to adjust bending, sizes, colours, gap, etc.
  4. F12 renders a PNG with transparent background + black outlines (Freestyle),
     ready to drop into Illustrator to add wires, battery, ammeter and labels.
"""

import bpy
import bmesh
import math

# ---------------------------------------------------------------- parameters
L = 12.0            # strip length
W = 2.0             # strip width
T_BRONZE = 0.25     # bronze substrate thickness
T_PI = 0.08         # polyimide thickness
T_AU = 0.015        # gold electrode thickness
BEND_RADIUS = 30.0  # radius of curvature (larger = flatter, 1e9 = flat)
GAP = 0.04          # gap between the two electrode tips (0 = continuous bridge)
PAD_HALF_W = 0.8    # half width of electrode pads
TIP_HALF_W = 0.05   # half width of electrode tip
TAPER_LEN = 1.5     # length of the tapered part
PAD_END = L / 2 - 0.4   # pads stop 0.4 before the strip ends (follows L)
N = 240             # segments along the length (smoothness of the bend)

ROD_R = 0.35
ROD_LEN = W + 1.0     # rods stick out 0.5 on each side (follows W)
ROD_X = L / 2 - 0.3     # holding rods sit 0.3 from the strip ends (follows L)
WIRE_POSTS = True   # thin vertical wires from each pad (to connect circuit in Illustrator)
ORTHO_CAMERA = True

SINK = 0.005        # layers overlap slightly so Freestyle doesn't draw dashed lines

# under-etched cavity in the polyimide below the constriction (suspended bridge)
CAVITY = True
CAVITY_RX = 0.55    # half length along the strip
CAVITY_RY = 0.40    # half width across the strip
CAVITY_DEPTH = 0.07 # must stay below T_PI so the bronze is not exposed
COL_CAVITY = (0.06, 0.01, 0.005)  # dark inside of the cavity

CAM_ELEVATION = 18  # degrees above the horizon (0 = pure side view, 90 = top view)
CAM_DISTANCE = 20.0

# colours are LINEAR values (what Blender's colour picker shows as RGB, not hex)
COL_BRONZE = (0.16, 0.065, 0.025)
COL_PI     = (0.33, 0.055, 0.03)
COL_GOLD   = (0.40, 0.25, 0.04)
COL_ROD    = (0.30, 0.30, 0.30)
COL_WIRE   = (0.01, 0.01, 0.01)


# ---------------------------------------------------------------- helpers
def bend(x, y, z):
    """Wrap the flat (x, z) plane around a cylinder of radius BEND_RADIUS (axis along y)."""
    th = x / BEND_RADIUS
    r = BEND_RADIUS + z
    return (r * math.sin(th), y, r * math.cos(th) - BEND_RADIUS)


def make_material(name, rgb, metallic=0.0, roughness=0.5):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*rgb, 1.0)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = roughness
    m.diffuse_color = (*rgb, 1.0)  # colour in Solid viewport mode
    return m


def link(obj):
    bpy.context.collection.objects.link(obj)
    return obj


def slab(name, x0, x1, half_width, z0, z1, mat, n=N):
    """Bent slab from x0..x1, width given by half_width(x), thickness z0..z1."""
    verts, faces = [], []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        w = half_width(x)
        for (y, z) in ((-w, z0), (w, z0), (w, z1), (-w, z1)):
            verts.append(bend(x, y, z))
    for i in range(n):
        for k in range(4):
            a, b = 4 * i + k, 4 * i + (k + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    faces.append((3, 2, 1, 0))
    e = 4 * n
    faces.append((e, e + 1, e + 2, e + 3))

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


def smooth(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for op in ("shade_auto_smooth", "shade_smooth_by_angle"):
        try:
            getattr(bpy.ops.object, op)()
            return
        except Exception:
            pass
    bpy.ops.object.shade_smooth()
    try:
        obj.data.use_auto_smooth = True
    except Exception:
        pass


def rod(name, x, z_local, mat):
    loc = bend(x, 0.0, z_local)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=ROD_R, depth=ROD_LEN,
                                        location=loc, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    smooth(o)
    return o


# ---------------------------------------------------------------- clear scene
if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
scene = bpy.context.scene

m_bronze = make_material("Bronze", COL_BRONZE, 0.3, 0.55)
m_pi = make_material("Polyimide", COL_PI, 0.0, 0.45)
m_gold = make_material("Gold", COL_GOLD, 0.5, 0.4)
m_rod = make_material("Rod", COL_ROD, 0.3, 0.4)
m_wire = make_material("Wire", COL_WIRE, 0.0, 0.6)

# ---------------------------------------------------------------- layers
full = lambda x: W / 2
slab("Bronze_substrate", -L / 2, L / 2, full, -T_BRONZE, 0.0, m_bronze)
# polyimide is a hair narrower/shorter than the bronze so no faces are coplanar (avoids ragged lines)
EPS = 0.004
polyimide = slab("Polyimide", -L / 2 + EPS, L / 2 - EPS, lambda x: W / 2 - EPS, -SINK, T_PI, m_pi)

if CAVITY:
    # ellipsoid cutter centred on the top surface at the gap; its material ends up on the cavity walls
    m_cav = make_material("Cavity", COL_CAVITY, 0.0, 0.8)
    cx, cy, cz = bend(0.0, 0.0, T_PI)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0, location=(cx, cy, cz))
    cutter = bpy.context.active_object
    cutter.scale = (CAVITY_RX, CAVITY_RY, CAVITY_DEPTH)
    cutter.data.materials.append(m_cav)
    polyimide.data.materials.append(m_cav)
    mod = polyimide.modifiers.new("Cavity", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    try:
        mod.solver = "EXACT"
        mod.material_mode = "TRANSFER"
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = polyimide
    polyimide.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def pad_width(x):
    d = abs(x) - GAP / 2          # distance from tip
    if d >= TAPER_LEN:
        return PAD_HALF_W
    return TIP_HALF_W + (PAD_HALF_W - TIP_HALF_W) * d / TAPER_LEN


z0, z1 = T_PI - SINK, T_PI + T_AU
slab("Electrode_left", -PAD_END, -GAP / 2, pad_width, z0, z1, m_gold, n=N // 2)
slab("Electrode_right", GAP / 2, PAD_END, pad_width, z0, z1, m_gold, n=N // 2)

# ---------------------------------------------------------------- rods
top = T_PI + T_AU + ROD_R
rod("Rod_left", -ROD_X, top, m_rod)
rod("Rod_right", ROD_X, top, m_rod)
rod("Pushing_rod", 0.0, -T_BRONZE - ROD_R, m_rod)

# ---------------------------------------------------------------- wire posts
if WIRE_POSTS:
    for name, x in (("Wire_left", -L / 4), ("Wire_right", L / 4)):
        bx, by, bz = bend(x, 0.0, z1)
        h = 2.0
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.02, depth=h,
                                            location=(bx, by, bz + h / 2))
        o = bpy.context.active_object
        o.name = name
        o.data.materials.append(m_wire)

# ---------------------------------------------------------------- camera
target = link(bpy.data.objects.new("Camera_target", None))
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
if ORTHO_CAMERA:
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = L * 1.125 + 2 * ROD_R   # zoom follows the strip length
cam = link(bpy.data.objects.new("Camera", cam_data))
el = math.radians(CAM_ELEVATION)
cam.location = (0.0, -CAM_DISTANCE * math.cos(el), CAM_DISTANCE * math.sin(el))  # frontal view
c = cam.constraints.new("TRACK_TO")
c.target = target
c.track_axis = "TRACK_NEGATIVE_Z"
c.up_axis = "UP_Y"
scene.camera = cam

# ---------------------------------------------------------------- lights & world
sun = link(bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")))
sun.data.energy = 2.0
sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))

world = scene.world or bpy.data.worlds.new("World")
scene.world = world
try:
    world.use_nodes = True
except Exception:
    pass
bg = world.node_tree.nodes.get("Background")
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 0.45

# ---------------------------------------------------------------- render settings
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        scene.render.engine = eng
        break
    except TypeError:
        pass
scene.render.resolution_x = 2400
scene.render.resolution_y = 1200
scene.render.film_transparent = True   # transparent PNG for Illustrator
scene.render.use_freestyle = True      # black outlines, illustration look
scene.render.line_thickness_mode = "ABSOLUTE"
scene.render.line_thickness = 1.2
fs = bpy.context.view_layer.freestyle_settings
ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("LineSet")
ls.select_by_visibility = True
ls.visibility = "VISIBLE"
ls.select_silhouette = True
ls.select_border = True
ls.select_crease = True
ls.select_material_boundary = True   # line where gold meets polyimide, polyimide meets bronze
fs.crease_angle = math.radians(120)
try:
    scene.view_settings.view_transform = "Standard"  # true colours, not Filmic/AgX
except Exception:
    pass

print("Schematic built. Press F12 to render, or Numpad 0 to look through the camera.")
