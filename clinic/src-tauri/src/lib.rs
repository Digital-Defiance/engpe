//! Tauri IPC for the offline ENGPE clinic.

use engpe_native::{evaluate_clinic_sweep_inner, ClinicSweepResult};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScoreCellDto {
    patient: u32,
    disease: u32,
    plaintext: f64,
    decoded: f64,
    abs_error: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepResultDto {
    patient_count: u32,
    disease_count: u32,
    backend: String,
    ntt_ms: f64,
    max_abs_error: f64,
    scores: Vec<ScoreCellDto>,
    airgapped: bool,
}

#[tauri::command]
fn evaluate_clinic_sweep(
    patient_count: u32,
    disease_count: u32,
    seed: u32,
) -> Result<SweepResultDto, String> {
    let inner = evaluate_clinic_sweep_inner(patient_count, disease_count, seed)?;
    Ok(to_dto(inner))
}

fn to_dto(inner: ClinicSweepResult) -> SweepResultDto {
    SweepResultDto {
        patient_count: inner.patient_count,
        disease_count: inner.disease_count,
        backend: inner.backend,
        ntt_ms: inner.ntt_ms,
        max_abs_error: inner.max_abs_error,
        airgapped: inner.airgapped,
        scores: inner
            .scores
            .into_iter()
            .map(|s| ScoreCellDto {
                patient: s.patient,
                disease: s.disease,
                plaintext: s.plaintext,
                decoded: s.decoded,
                abs_error: s.abs_error,
            })
            .collect(),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![evaluate_clinic_sweep])
        .run(tauri::generate_context!())
        .expect("error while running ENGPE clinic");
}
