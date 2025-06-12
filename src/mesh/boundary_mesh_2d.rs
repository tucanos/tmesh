//! Boundary of `Mesh2d`
use super::{Edge, Mesh, Node};
use crate::{Tag, Vert2d};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Edge mesh in 2d
pub struct BoundaryMesh2d {
    verts: Vec<Vert2d>,
    elems: Vec<Edge>,
    etags: Vec<Tag>,
    faces: Vec<Node>,
    ftags: Vec<Tag>,
}

impl BoundaryMesh2d {
    /// Create a new mesh from coordinates, connectivities and tags
    pub fn new(
        verts: Vec<Vert2d>,
        elems: Vec<Edge>,
        etags: Vec<Tag>,
        faces: Vec<Node>,
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
}

impl Mesh<2, 2, 1> for BoundaryMesh2d {
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

    fn vert(&self, i: usize) -> &Vert2d {
        &self.verts[i]
    }

    fn verts(&self) -> impl ExactSizeIterator<Item = &Vert2d> + Clone + '_ {
        self.verts.iter()
    }

    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vert2d> + '_ {
        self.verts.iter_mut()
    }

    fn par_verts(
        &self,
    ) -> impl rayon::prelude::IndexedParallelIterator<Item = &Vert2d> + Clone + '_ {
        self.verts.par_iter()
    }

    fn add_verts<I: ExactSizeIterator<Item = Vert2d>>(&mut self, v: I) {
        self.verts.extend(v);
    }

    fn n_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem(&self, i: usize) -> &Edge {
        &self.elems[i]
    }

    fn elems(&self) -> impl ExactSizeIterator<Item = &Edge> + Clone + '_ {
        self.elems.iter()
    }

    fn elems_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Edge> + '_ {
        self.elems.iter_mut()
    }

    fn par_elems(&self) -> impl IndexedParallelIterator<Item = &Edge> + Clone + '_ {
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

    fn add_elems<I1: ExactSizeIterator<Item = Edge>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Edge, Tag)>>(&mut self, elems_and_tags: I) {
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

    fn face(&self, i: usize) -> &Node {
        &self.faces[i]
    }

    fn faces(&self) -> impl ExactSizeIterator<Item = &Node> + Clone + '_ {
        self.faces.iter()
    }

    fn faces_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Node> + '_ {
        self.faces.iter_mut()
    }

    fn par_faces(&self) -> impl IndexedParallelIterator<Item = &Node> + Clone + '_ {
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

    fn add_faces<I1: ExactSizeIterator<Item = Node>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Node, Tag)>>(&mut self, faces_and_tags: I) {
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
        mesh::{rectangle_mesh, Edge, Mesh, Mesh2d, Simplex},
        Vert2d,
    };
    use rayon::iter::ParallelIterator;

    use super::BoundaryMesh2d;

    #[test]
    fn test_rectangle() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 20);

        let (mut bdy, ids): (BoundaryMesh2d, _) = msh.boundary();

        let faces = bdy.compute_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 4);
        bdy.check(&faces).unwrap();

        assert_eq!(bdy.n_verts(), 2 * 10 + 2 * 20 - 4);
        assert_eq!(bdy.n_elems(), 2 * 9 + 2 * 19);

        for (i, &j) in ids.iter().enumerate() {
            let pi = *bdy.vert(i);
            let pj = *msh.vert(j);
            let d = (pj - pi).norm();
            assert!(d < 1e-12);
        }
    }

    #[test]
    fn test_integrate() {
        let v0 = Vert2d::new(0.0, 0.0);
        let v1 = Vert2d::new(0.5, 0.0);
        let ge = [&v0, &v1];
        assert_delta!(Edge::vol(ge), 0.5, 1e-12);
        let ge = [&v1, &v0];
        assert_delta!(Edge::vol(ge), 0.5, 1e-12);

        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let tag = 1;
        let (bdy, ids): (BoundaryMesh2d, _) = msh.extract_faces(|t| t == tag);
        let f_bdy = ids.iter().map(|&i| f[i]).collect::<Vec<_>>();

        let val = bdy.integrate(&f_bdy, |_| 1.0);
        assert_delta!(val, 1.0, 1e-12);

        let val = bdy.integrate(&f_bdy, |x| x);
        assert_delta!(val, 0.5, 1e-12);

        let nrm = bdy.norm(&f_bdy);
        assert_delta!(nrm, 1.0 / 3.0_f64.sqrt(), 1e-12);
    }
}
