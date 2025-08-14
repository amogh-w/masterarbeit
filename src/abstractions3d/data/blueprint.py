from abstractions3d.dsl.nodes import Rect3D, Move3D, Union3D, SymRef3D

# ---------------- Table ----------------
def Table3D(
    top_length, top_depth, top_thickness,
    leg_length, leg_depth, leg_height,
):
    margin = 0.05
    leg_offset_x = top_length / 2 - leg_length / 2 - margin
    leg_offset_z = top_depth / 2 - leg_depth / 2 - margin

    # Tabletop centered above legs
    tabletop = Move3D(
        Rect3D(top_length, top_thickness, top_depth),  # x=length, y=height, z=depth
        0.0, leg_height + top_thickness / 2.0, 0.0
    )

    # One leg (front-right)
    leg = Move3D(
        Rect3D(leg_length, leg_height, leg_depth),
        leg_offset_x, leg_height / 2.0, leg_offset_z
    )

    legs_x = SymRef3D(leg, "x")
    legs_xz = SymRef3D(legs_x, "z")

    return Union3D(tabletop, legs_xz)


# ---------------- Chair ----------------
def Chair3D(
    seat_length, seat_depth, seat_thickness,
    leg_length, leg_depth, leg_height,
    backrest_height, backrest_thickness
):
    margin = 0.05
    leg_offset_x = seat_length / 2 - leg_length / 2 - margin
    leg_offset_z = seat_depth / 2 - leg_depth / 2 - margin

    # Seat
    seat = Move3D(
        Rect3D(seat_length, seat_thickness, seat_depth),
        0.0, leg_height + seat_thickness / 2.0, 0.0
    )

    # Legs
    leg = Move3D(
        Rect3D(leg_length, leg_height, leg_depth),
        leg_offset_x, leg_height / 2.0, leg_offset_z
    )
    legs_x = SymRef3D(leg, "x")
    legs_xz = SymRef3D(legs_x, "z")

    # Backrest (at back of seat along z)
    backrest = Move3D(
        Rect3D(seat_length, backrest_height, backrest_thickness),
        0.0,
        leg_height + seat_thickness + backrest_height / 2.0,
        -(seat_depth / 2 - backrest_thickness / 2)
    )

    return Union3D(seat, Union3D(legs_xz, backrest))


# ---------------- Bench ----------------
def Bench3D(
    seat_length, seat_depth, seat_thickness,
    leg_length, leg_depth, leg_height,
    backrest_height=0.0, backrest_thickness=0.0
):
    margin = 0.05
    leg_offset_x = seat_length / 2 - leg_length / 2 - margin
    leg_offset_z = seat_depth / 2 - leg_depth / 2 - margin

    # Seat
    seat = Move3D(
        Rect3D(seat_length, seat_thickness, seat_depth),
        0.0, leg_height + seat_thickness / 2.0, 0.0
    )

    # Legs
    leg = Move3D(
        Rect3D(leg_length, leg_height, leg_depth),
        leg_offset_x, leg_height / 2.0, leg_offset_z
    )
    legs_x = SymRef3D(leg, "x")
    legs_xz = SymRef3D(legs_x, "z")

    # Optional backrest
    if backrest_height > 0 and backrest_thickness > 0:
        backrest = Move3D(
            Rect3D(seat_length, backrest_height, backrest_thickness),
            0.0,
            leg_height + seat_thickness + backrest_height / 2.0,
            -(seat_depth / 2 - backrest_thickness / 2)
        )
        return Union3D(seat, Union3D(legs_xz, backrest))

    return Union3D(seat, legs_xz)