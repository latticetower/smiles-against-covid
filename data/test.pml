load target.pdb
fetch 6woj
set cartoon_transparency, 0.5, 6woj
select macrodom, model 6woj and chain B
select macrodom2, model 6woj and chain C
remove solvent
hide everything, macrodom
hide everything, macrodom2
hide everything, target

select ligand, model 6woj and organic
show sticks, ligand
set cartoon_transparency, 0, ligand
set stick_radius, 0.5
set stick_color, red

align macrodom, target

show surface, macrodom2
show surface, target

set surface_color, white, macrodom2
set transparency, 0.3, target

run hydro_code.py
select target
hydrophobicity("(sele)", 1)
set_view (\
    -0.066783518,    0.982642472,   -0.173060924,\
    -0.321917117,    0.142952368,    0.935911357,\
     0.944409013,    0.118216828,    0.306783557,\
     0.000000000,    0.000000000, -138.016525269,\
    16.918731689,   14.838355064,    6.542036057,\
   108.813278198,  167.219772339,  -20.000000000 )
### cut above here and paste into script ###
save target_with_6woj.pdb

save_images("(sele)")
