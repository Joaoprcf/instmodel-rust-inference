use instmodel_inference::activation::Activation;
use instmodel_inference::gpu::{GpuModel, GpuModelError, PopulationPack};
use instmodel_inference::graph::{Graph, ModelGraph};
use instmodel_inference::params::ParamLayout;

/// Non-empty params region (normalize constants) plus two weight slots.
fn demo_graph() -> ModelGraph {
    let graph = Graph::new();
    let x = graph.input(4, None);
    let normalized = graph.normalize(&x, vec![1.0; 4], vec![2.0; 4]);
    let hidden = graph.dense(&normalized, 3, Some(Activation::Tanh));
    let output = graph.dense(&hidden, 2, None);
    graph.model(vec![&x], &output)
}

fn sentinel_theta(len: usize) -> Vec<f32> {
    (0..len).map(|k| k as f32).collect()
}

#[test]
fn candidate_theta_matches_flatten_from_info() {
    let model_graph = demo_graph();
    let layout = model_graph.weights_map().unwrap();
    let theta = sentinel_theta(layout.total);
    let info = model_graph.compile(&theta).unwrap();

    let pack = PopulationPack::new(&info, 3).unwrap();
    assert_eq!(pack.theta_len(), layout.total);
    let canonical = ParamLayout::flatten_from_info(&info);
    assert_eq!(canonical, theta);
    for candidate in 0..3 {
        assert_eq!(pack.candidate_theta(candidate).unwrap(), &theta[..]);
    }
}

#[test]
fn write_candidate_targets_only_its_candidate() {
    let model_graph = demo_graph();
    let info = model_graph.compile_zeroed().unwrap();
    let mut pack = PopulationPack::new(&info, 3).unwrap();

    let template = GpuModel::from_info(&info).unwrap();
    let theta = sentinel_theta(pack.theta_len());
    pack.write_candidate(1, &theta).unwrap();

    assert_eq!(pack.candidate_theta(0).unwrap(), vec![0.0; theta.len()]);
    assert_eq!(pack.candidate_theta(1).unwrap(), &theta[..]);
    assert_eq!(pack.candidate_theta(2).unwrap(), vec![0.0; theta.len()]);

    // The non-theta parts of every candidate stay identical to the template:
    // header + instructions before the weights region, params region after.
    let stride = pack.model_stride();
    let weights_start = template.weights_offset();
    let weights_end = weights_start + template.theta_len();
    for candidate in 0..3 {
        let blob = &pack.host_blob()[candidate * stride..(candidate + 1) * stride];
        assert_eq!(
            &blob[..weights_start],
            &template.as_f32_slice()[..weights_start],
            "candidate {candidate} header/instructions changed"
        );
        assert_eq!(
            &blob[weights_end..],
            &template.as_f32_slice()[weights_end..],
            "candidate {candidate} params region changed"
        );
    }
}

#[test]
fn write_candidate_f64_narrows_values() {
    let model_graph = demo_graph();
    let info = model_graph.compile_zeroed().unwrap();
    let mut pack = PopulationPack::new(&info, 2).unwrap();

    let theta_f64: Vec<f64> = (0..pack.theta_len()).map(|k| k as f64 * 0.25).collect();
    pack.write_candidate_f64(0, &theta_f64).unwrap();

    let expected: Vec<f32> = theta_f64.iter().map(|&v| v as f32).collect();
    assert_eq!(pack.candidate_theta(0).unwrap(), &expected[..]);
}

#[test]
fn written_candidate_matches_directly_serialized_model() {
    // Writing theta into a zeroed pack must equal serializing compile(theta).
    let model_graph = demo_graph();
    let layout = model_graph.weights_map().unwrap();
    let theta = sentinel_theta(layout.total);

    let mut pack = PopulationPack::from_graph(&model_graph, 2).unwrap();
    pack.write_candidate(1, &theta).unwrap();

    let direct = GpuModel::from_info(&model_graph.compile(&theta).unwrap()).unwrap();
    let stride = pack.model_stride();
    let candidate_blob = &pack.host_blob()[stride..2 * stride];
    assert_eq!(candidate_blob, direct.as_f32_slice());
}

#[test]
fn from_graph_starts_zeroed_and_tiles_the_template() {
    let model_graph = demo_graph();
    let info = model_graph.compile_zeroed().unwrap();
    let template = GpuModel::from_info(&info).unwrap();
    let pack = PopulationPack::from_graph(&model_graph, 2).unwrap();

    assert_eq!(pack.n_candidates(), 2);
    assert_eq!(pack.theta_len(), model_graph.weights_map().unwrap().total);
    assert_eq!(pack.host_blob().len(), 2 * template.full_size());
    assert_eq!(pack.as_bytes().len(), 2 * template.full_size() * 4);
    // Normalize constants live in a params region after the weights region.
    assert!(info.parameters.is_some());
    assert!(template.weights_offset() + pack.theta_len() < pack.model_stride());
    for candidate in 0..2 {
        assert!(
            pack.candidate_theta(candidate)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }
}

#[test]
fn error_paths() {
    let model_graph = demo_graph();
    let info = model_graph.compile_zeroed().unwrap();

    assert!(matches!(
        PopulationPack::new(&info, 0),
        Err(GpuModelError::EmptyPopulation)
    ));

    let mut pack = PopulationPack::new(&info, 2).unwrap();
    let theta = sentinel_theta(pack.theta_len());

    assert!(matches!(
        pack.write_candidate(2, &theta),
        Err(GpuModelError::CandidateOutOfBounds { index: 2, count: 2 })
    ));
    assert!(matches!(
        pack.write_candidate(0, &theta[..theta.len() - 1]),
        Err(GpuModelError::ThetaLengthMismatch { .. })
    ));
    assert!(matches!(
        pack.write_candidate_f64(0, &[1.0f64]),
        Err(GpuModelError::ThetaLengthMismatch { .. })
    ));
    assert!(matches!(
        pack.candidate_theta(5),
        Err(GpuModelError::CandidateOutOfBounds { index: 5, count: 2 })
    ));
}
