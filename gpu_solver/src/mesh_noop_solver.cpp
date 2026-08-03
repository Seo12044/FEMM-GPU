// Mesh-only FEMM helper: deliberately performs no magnetic solve.
//
// A private, staged copy of stock FEMM invokes this executable under the
// expected fkn.exe name after Triangle has written the mesh.  Returning zero
// leaves the mesh files intact for the GPU artifact converter.  It must never
// be installed over or registered with stock FEMM.
int main(int, char**)
{
  return 0;
}
