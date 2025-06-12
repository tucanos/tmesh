//! Boundary of `Mesh3d`
use super::{Edge, Mesh, Triangle};
use crate::{Result, Tag, Vert3d};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fs::OpenOptions;

/// Triangle mesh in 3d
pub struct BoundaryMesh3d {
    verts: Vec<Vert3d>,
    elems: Vec<Triangle>,
    etags: Vec<Tag>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
}

impl BoundaryMesh3d {
    /// Create a new mesh from coordinates, connectivities and tags
    pub fn new(
        verts: Vec<Vert3d>,
        elems: Vec<Triangle>,
        etags: Vec<Tag>,
        faces: Vec<Edge>,
        ftags: Vec<Tag>,
    ) -> Self {
        Self {
            verts,
            elems,
            etags,
            faces,
            ftags,
        }
    }

    /// Read a stl file
    pub fn read_stl(file_name: &str) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
        let stl = stl_io::read_stl(&mut file).unwrap();

        let mut verts = Vec::with_capacity(stl.vertices.len());
        verts.extend(
            stl.vertices
                .iter()
                .map(|v| Vert3d::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))),
        );

        let mut elems = Vec::with_capacity(3 * stl.faces.len());
        elems.extend(stl.faces.iter().map(|v| v.vertices));
        let etags = vec![1; stl.faces.len()];
        let faces = Vec::new();
        let ftags = Vec::new();

        Ok(Self::new(verts, elems, etags, faces, ftags))
    }
}

impl Mesh<3, 3, 2> for BoundaryMesh3d {
    fn empty() -> Self {
        Self {
            verts: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            faces: Vec::new(),
            ftags: Vec::new(),
        }
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> &Vert3d {
        &self.verts[i]
    }

    fn verts(&self) -> impl ExactSizeIterator<Item = &Vert3d> + Clone + '_ {
        self.verts.iter()
    }

    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vert3d> + '_ {
        self.verts.iter_mut()
    }

    fn par_verts(
        &self,
    ) -> impl rayon::prelude::IndexedParallelIterator<Item = &Vert3d> + Clone + '_ {
        self.verts.par_iter()
    }

    fn add_verts<I: ExactSizeIterator<Item = Vert3d>>(&mut self, v: I) {
        self.verts.extend(v);
    }

    fn n_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem(&self, i: usize) -> &Triangle {
        &self.elems[i]
    }

    fn elems(&self) -> impl ExactSizeIterator<Item = &Triangle> + Clone + '_ {
        self.elems.iter()
    }

    fn elems_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Triangle> + '_ {
        self.elems.iter_mut()
    }

    fn par_elems(&self) -> impl IndexedParallelIterator<Item = &Triangle> + Clone + '_ {
        self.elems.par_iter()
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.etags.iter().cloned()
    }

    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.etags.iter_mut()
    }

    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.etags.par_iter().cloned()
    }

    fn add_elems<I1: ExactSizeIterator<Item = Triangle>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    ) {
        self.elems.extend(elems);
        self.etags.extend(etags);
    }

    fn clear_elems(&mut self) {
        self.elems.clear();
        self.etags.clear();
    }

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Triangle, Tag)>>(
        &mut self,
        elems_and_tags: I,
    ) {
        self.elems.reserve(elems_and_tags.len());
        self.etags.reserve(elems_and_tags.len());
        for (e, t) in elems_and_tags {
            self.elems.push(e);
            self.etags.push(t);
        }
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> &Edge {
        &self.faces[i]
    }

    fn faces(&self) -> impl ExactSizeIterator<Item = &Edge> + Clone + '_ {
        self.faces.iter()
    }

    fn faces_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Edge> + '_ {
        self.faces.iter_mut()
    }

    fn par_faces(&self) -> impl IndexedParallelIterator<Item = &Edge> + Clone + '_ {
        self.faces.par_iter()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.ftags.iter().cloned()
    }

    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.ftags.iter_mut()
    }

    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.ftags.par_iter().cloned()
    }

    fn add_faces<I1: ExactSizeIterator<Item = Edge>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        faces: I1,
        ftags: I2,
    ) {
        self.faces.extend(faces);
        self.ftags.extend(ftags);
    }

    fn clear_faces(&mut self) {
        self.faces.clear();
        self.ftags.clear();
    }

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Edge, Tag)>>(&mut self, faces_and_tags: I) {
        self.faces.reserve(faces_and_tags.len());
        self.ftags.reserve(faces_and_tags.len());
        for (e, t) in faces_and_tags {
            self.faces.push(e);
            self.ftags.push(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        mesh::{box_mesh, BoundaryMesh3d, Mesh, Mesh3d, Simplex, Triangle},
        Vert3d,
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();

        let faces = bdy.compute_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 12);
        bdy.check(&faces).unwrap();

        let vol = bdy.gelems().map(Triangle::vol).sum::<f64>();
        assert_delta!(vol, 10.0, 1e-12);
    }

    #[test]
    fn test_integrate() {
        let v0 = Vert3d::new(0.0, 0.0, 1.0);
        let v1 = Vert3d::new(0.5, 0.0, 1.0);
        let v2 = Vert3d::new(0.0, 0.5, 1.0);
        let ge = [&v0, &v1, &v2];
        assert_delta!(Triangle::vol(ge), 0.125, 1e-12);
        let ge = [&v1, &v0, &v2];
        assert_delta!(Triangle::vol(ge), 0.125, 1e-12);

        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let tag = 3;
        let (bdy, ids): (BoundaryMesh3d, _) = msh.extract_faces(|t| t == tag);
        let f_bdy = ids.iter().map(|&i| f[i]).collect::<Vec<_>>();

        let val = bdy.integrate(&f_bdy, |_| 1.0);
        assert_delta!(val, 1.0, 1e-12);

        let val = bdy.integrate(&f_bdy, |x| x);
        assert_delta!(val, 0.5, 1e-12);

        let nrm = bdy.norm(&f_bdy);
        assert_delta!(nrm, 1.0 / 3.0_f64.sqrt(), 1e-12);
    }
}
