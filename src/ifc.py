# create ifc elements

import math
import uuid

import ifcopenshell
import numpy as np

from src.geometry import get_corner, get_oriented_bbox, sq_distance


def create_guid():
    return ifcopenshell.guid.compress(uuid.uuid1().hex)


# create a new blank ifc file
def setup_ifc_file(blueprint):
    ifc = ifcopenshell.open(blueprint)
    ifcNew = ifcopenshell.file(schema=ifc.schema)

    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]

    ifcNew.add(project)
    ifcNew.add(owner_history)
    ifcNew.add(context)
    ifcNew.add(floor)

    #Add other spatial structure elements and relationships so items are properly referenced
    for site in ifc.by_type("IfcSite"):
        ifcNew.add(site)
    for bldg in ifc.by_type("IfcBuilding"):
        ifcNew.add(bldg)
    for rel in ifc.by_type("IfcRelAggregates"):
        ifcNew.add(rel)

    return ifcNew


# create an IFC beam with a susbtracted section for a tee
def CreatePartialBeam(
    ifcFile, container, name, primary_beam, secondary_beam, owner_history, context
):
    Z = 0.0, 0.0, 1.0
    X = 1.0, 0.0, 0.0

    F1 = ifcFile.createIfcPipeFitting(create_guid(), owner_history, name)
    F1.ObjectType = "beam"

    F1_Point = ifcFile.createIfcCartesianPoint((0.0, 0.0, 0.0))
    F1_Axis2Placement = ifcFile.createIfcAxis2Placement3D(F1_Point)
    F1_Axis2Placement.Axis = ifcFile.createIfcDirection(Z)
    F1_Axis2Placement.RefDirection = ifcFile.createIfcDirection(X)

    F1_Placement = ifcFile.createIfcLocalPlacement(
        container.ObjectPlacement, F1_Axis2Placement
    )
    F1.ObjectPlacement = F1_Placement

    boolean_result = ifcFile.createIfcBooleanResult(
        "DIFFERENCE", secondary_beam, primary_beam
    )
    CSGSolid = ifcFile.createIfcCsgSolid(boolean_result)

    F1_Repr = ifcFile.createIfcShapeRepresentation()
    F1_Repr.ContextOfItems = context
    F1_Repr.RepresentationIdentifier = "Body"
    F1_Repr.RepresentationType = "CSGSolid"
    F1_Repr.Items = [CSGSolid]

    F1_DefShape = ifcFile.createIfcProductDefinitionShape()
    F1_DefShape.Representations = [F1_Repr]
    F1.Representation = F1_DefShape

    Flr1_Container = ifcFile.createIfcRelContainedInSpatialStructure(
        create_guid(), owner_history
    )
    Flr1_Container.RelatedElements = [F1]
    Flr1_Container.RelatingStructure = container


# create an IFC BEAM
def CreateBeam(
    ifcFile,
    container,
    name,
    section,
    L,
    position,
    direction,
    owner_history,
    context,
    colour=None,
):
    Z = 0.0, 0.0, 1.0
    # print('length', L)
    B1 = ifcFile.createIfcPipeFitting(create_guid(), owner_history, name)
    B1.ObjectType = "beam"

    # print(type(position[0]))
    B1_Point = ifcFile.createIfcCartesianPoint(tuple(position))
    # B1_Point =ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) )
    B1_Axis2Placement = ifcFile.createIfcAxis2Placement3D(B1_Point)
    B1_Axis2Placement.Axis = ifcFile.createIfcDirection(direction)
    B1_Axis2Placement.RefDirection = ifcFile.createIfcDirection(
        np.cross(direction, Z).tolist()
    )

    B1_Placement = ifcFile.createIfcLocalPlacement(
        container.ObjectPlacement, B1_Axis2Placement
    )
    B1.ObjectPlacement = B1_Placement
    # B1Point = ifcFile.createIfcCartesianPoint ( tuple(position) )
    B1Point = ifcFile.createIfcCartesianPoint((0.0, 0.0, 0.0))
    B1_ExtrudePlacement = ifcFile.createIfcAxis2Placement3D(B1Point)
    # print (B1Point, B1_ExtrudePlacement, B1_Placement)

    B1_Extruded = ifcFile.createIfcExtrudedAreaSolid()
    B1_Extruded.SweptArea = section
    B1_Extruded.Position = B1_ExtrudePlacement
    B1_Extruded.ExtrudedDirection = ifcFile.createIfcDirection(Z)
    B1_Extruded.Depth = L

    # add colour
    if colour is not None:
        shade = ifcFile.createIfcSurfaceStyleRendering(colour)
        surfaceStyle = ifcFile.createIfcSurfaceStyle(colour.Name, "BOTH", (shade,))
        presStyleAssign = ifcFile.createIfcPresentationStyleAssignment((surfaceStyle,))
        ifcFile.createIfcStyledItem(B1_Extruded, (presStyleAssign,), colour.Name)

    B1_Repr = ifcFile.createIfcShapeRepresentation()
    B1_Repr.ContextOfItems = context
    B1_Repr.RepresentationIdentifier = "Body"
    B1_Repr.RepresentationType = "SweptSolid"
    B1_Repr.Items = [B1_Extruded]

    B1_DefShape = ifcFile.createIfcProductDefinitionShape()
    B1_DefShape.Representations = [B1_Repr]
    B1.Representation = B1_DefShape

    Flr1_Container = ifcFile.createIfcRelContainedInSpatialStructure(
        create_guid(), owner_history
    )
    Flr1_Container.RelatedElements = [B1]
    Flr1_Container.RelatingStructure = container


# create the geometric representation of an IFC BEAM


def CreateBeamGeom(ifcFile, section, L, position, direction):
    Z = 0.0, 0.0, 1.0
    # print('length', L)

    # B1Point = ifcFile.createIfcCartesianPoint ( tuple(position) )
    B1Point = ifcFile.createIfcCartesianPoint(tuple(position))
    B1_ExtrudePlacement = ifcFile.createIfcAxis2Placement3D(B1Point)
    B1_ExtrudePlacement.Axis = ifcFile.createIfcDirection(direction)
    B1_ExtrudePlacement.RefDirection = ifcFile.createIfcDirection(
        np.cross(direction, Z).tolist()
    )
    # print (B1Point, B1_ExtrudePlacement, B1_Placement)

    B1_Extruded = ifcFile.createIfcExtrudedAreaSolid()
    B1_Extruded.SweptArea = section
    B1_Extruded.Position = B1_ExtrudePlacement
    B1_Extruded.ExtrudedDirection = ifcFile.createIfcDirection(Z)
    B1_Extruded.Depth = L

    return B1_Extruded


# create an IFC elbow
def CreateElbow(
    ifcFile,
    container,
    name,
    section,
    a,
    x,
    y,
    axis_dir,
    position,
    direction,
    owner_history,
    context,
    Z,
    colour=None,
):
    X = 1.0, 0.0, 0.0
    # print('length', L)
    B1 = ifcFile.createIfcPipeFitting(create_guid(), owner_history, name)
    B1.ObjectType = "elbow"

    # print(type(position[0]))
    B1_Point = ifcFile.createIfcCartesianPoint(tuple(position))
    # B1_Point =ifcFile.createIfcCartesianPoint ( (0.0,0.0,0.0) )
    B1_Axis2Placement = ifcFile.createIfcAxis2Placement3D(B1_Point)
    B1_Axis2Placement.Axis = ifcFile.createIfcDirection(direction)
    B1_Axis2Placement.RefDirection = ifcFile.createIfcDirection(
        np.cross(direction, Z).tolist()
    )

    B1_Placement = ifcFile.createIfcLocalPlacement(
        container.ObjectPlacement, B1_Axis2Placement
    )
    B1.ObjectPlacement = B1_Placement
    # B1Point = ifcFile.createIfcCartesianPoint ( position )
    B1Point = ifcFile.createIfcCartesianPoint((0.0, 0.0, 0.0))
    B1_ExtrudePlacement = ifcFile.createIfcAxis2Placement3D(B1Point)
    # print (B1Point, B1_ExtrudePlacement, B1_Placement)

    B1Point2 = ifcFile.createIfcCartesianPoint((x, y, 0.0))
    B1_Axis1Placement = ifcFile.createIfcAxis1Placement(B1Point2)
    B1_Axis1Placement.Axis = ifcFile.createIfcDirection((axis_dir[0], axis_dir[1], 0.0))
    # B1_Axis1Placement.Axis = ifcFile.createIfcDirection((axis_dir[0], axis_dir[1], 0.))

    B1_Extruded = ifcFile.createIfcRevolvedAreaSolid()
    B1_Extruded.SweptArea = section
    B1_Extruded.Position = B1_ExtrudePlacement
    B1_Extruded.Axis = B1_Axis1Placement
    B1_Extruded.Angle = a

    # add colour
    if colour is not None:
        shade = ifcFile.createIfcSurfaceStyleRendering(colour)
        surfaceStyle = ifcFile.createIfcSurfaceStyle(colour.Name, "BOTH", (shade,))
        presStyleAssign = ifcFile.createIfcPresentationStyleAssignment((surfaceStyle,))
        ifcFile.createIfcStyledItem(B1_Extruded, (presStyleAssign,), colour.Name)

    B1_Repr = ifcFile.createIfcShapeRepresentation()
    B1_Repr.ContextOfItems = context
    B1_Repr.RepresentationIdentifier = "Body"
    B1_Repr.RepresentationType = "SweptSolid"
    B1_Repr.Items = [B1_Extruded]

    B1_DefShape = ifcFile.createIfcProductDefinitionShape()
    B1_DefShape.Representations = [B1_Repr]
    B1.Representation = B1_DefShape

    Flr1_Container = ifcFile.createIfcRelContainedInSpatialStructure(
        create_guid(), owner_history
    )
    Flr1_Container.RelatedElements = [B1]
    Flr1_Container.RelatingStructure = container


def Circle_Section(r, ifcfile, fill=False, vis_only=False):
    B1_Axis2Placement2D = ifcfile.createIfcAxis2Placement2D(
        ifcfile.createIfcCartesianPoint((0.0, 0.0, 0.0))
    )
    if fill:
        B1_AreaProfile = ifcfile.createIfcCircleProfileDef("AREA")
    else:
        B1_AreaProfile = ifcfile.createIfcCircleHollowProfileDef("AREA")
        if not vis_only:
            B1_AreaProfile.WallThickness = 2

    B1_AreaProfile.Position = B1_Axis2Placement2D
    B1_AreaProfile.Radius = r
    return B1_AreaProfile


def Rectangle_Section(ifcfile, x, y, fill=False):
    B1_Axis2Placement2D = ifcfile.createIfcAxis2Placement2D(
        ifcfile.createIfcCartesianPoint((0.0, 0.0, 0.0))
    )
    if fill:
        B1_AreaProfile = ifcfile.createIfcRectangleProfileDef("AREA")
    else:
        B1_AreaProfile = ifcfile.createIfcRectangleProfileDef("AREA")
        B1_AreaProfile.WallThickness = 2

    B1_AreaProfile.Position = B1_Axis2Placement2D
    B1_AreaProfile.XDim = x * 1000
    B1_AreaProfile.YDim = y * 1000
    return B1_AreaProfile


def draw_bbox(bbox, center, ifc, floor, owner_history, context):
    sectionC1 = Rectangle_Section(ifc, bbox[0], bbox[1], True)

    name = "bb"
    print("draw bbox", bbox, center)
    ConnectingBeam_1 = CreateBeam(
        ifc,
        container=floor,
        name=name,
        section=sectionC1,
        L=bbox[2] * 1000,
        position=([-bbox[2] * 500, 0.0, 0.0]),
        direction=(1.0, 0.0, 0.0),
        owner_history=owner_history,
        context=context,
        colour=None,
    )


def draw_cylinder(
    p1,
    p2,
    radius,
    colour,
    element_name1,
    element_name2,
    ifc,
    floor,
    owner_history,
    context,
):
    sectionC1 = Circle_Section(r=radius, ifcfile=ifc, vis_only=True)
    # print(p1, p2, radius)
    name = "rel " + element_name1 + " x " + element_name2
    ConnectingBeam_1 = CreateBeam(
        ifc,
        container=floor,
        name=name,
        section=sectionC1,
        L=math.sqrt(sq_distance(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])),
        position=([p2[0].item(), p2[1].item(), p2[2].item()]),
        direction=(
            (p1[0] - p2[0]).item(),
            (p1[1] - p2[1]).item(),
            (p1[2] - p2[2]).item(),
        ),
        owner_history=owner_history,
        context=context,
        colour=colour,
    )


# draw a visual indication of a topological relationship
def draw_relationship(
    element_name1,
    element1,
    element_name2,
    element2,
    ifc,
    floor,
    owner_history,
    context,
    colour=None,
):
    # define params
    # radius = 0.03
    radius_expansion = 1.3
    threshold = 0.1
    edge_distance = 0.1

    # get bboxes
    obb1 = get_oriented_bbox(element1)
    obb2 = get_oriented_bbox(element2)

    # if bboxes are roughly square, then use centerpoint,
    # otherwise use a point closer to the edge
    if obb1[1] < threshold:
        corner1 = obb1[3]
    else:
        corner1 = get_corner(obb1, obb2[3], edge_distance)
    if obb2[1] < threshold:
        corner2 = obb2[3]
    else:
        corner2 = get_corner(obb2, obb1[3], edge_distance)
    # print(corner1.X(),corner1.Y(),corner1.Z(),corner2.X(),corner2.Y(),corner1.Z())
    # print('rel', corner1, corner2, element_name1, element_name2)

    # dynamically scale radius
    radius = max(min(obb1[2]), min(obb2[2])) * radius_expansion

    draw_cylinder(
        corner1,
        corner2,
        radius,
        colour,
        element_name1,
        element_name2,
        ifc,
        floor,
        owner_history,
        context,
    )


#     draw_sphere(corner1, radius, colour, viewer)
#     draw_sphere(corner2, radius, 'red', viewer)
