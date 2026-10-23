i want to contribute this because flat ink inference retains completed blend
buffers when overlapping patches have empty raw input. the reproduction uses the
published `ink_9um` checkpoint on a crop from our PHerc0211 render.

on that crop, the fix releases two retained buffer pairs (256 KiB) before the final
flush in each depth direction. the before/after TIFF files are byte-identical, and
all 33 selected flat-inference CPU cases pass on current upstream main.

the saving is modest, but it removes unnecessary retention from a workflow our
project uses. this comment and the tested implementation were prepared with AI
assistance.
