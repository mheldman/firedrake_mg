SetFactory("OpenCASCADE");
Point(5) = { 0,  0, 0, 2.0};
Point(6) = { 2,  0, 0, .3};
Point(7) = {-2,  0, 0, .3};
Point(8) = { 0,  2, 0, .3};
Point(9) = { 0, -2, 0, .3};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};
Curve Loop(10) = {8, 5, 6, 7};
Plane Surface(2) = {10};
Physical Curve("Circle", 13) = {8, 7, 6, 5};
Physical Surface("Disc", 4) = {2};


