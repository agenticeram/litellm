//! The params stage of `_pre_call_checks`: while the live `litellm.drop_params` is off, a
//! deployment that does not support a param the request sends is dropped, unless that leaves
//! none of what the earlier stages kept. Then only the deployments that reject the param are
//! dropped, those whose call settings neither drop nor allow it. A group whose every deployment
//! rejects it raises one `UnsupportedParams`.

use std::collections::BTreeSet;

use litellm_python_compat::lenient::{LaxBool, ParamNames, TextEntries};
use serde::{Deserialize, de::value::MapDeserializer};
use serde_with::serde_as;

use super::PreCallFacts;
use crate::{error::Error, registry::Deployment};

/// The call settings `_deployment_rejects_params` reads. The request's win over the
/// deployment's `litellm_params`, one setting at a time.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct CallSettings {
    /// `drop_params`: every unsupported param is dropped.
    #[serde_as(as = "LaxBool")]
    pub drop_params: Option<bool>,
    /// `allowed_openai_params`: params passed on even where unsupported.
    #[serde_as(as = "Option<ParamNames>")]
    pub allowed_openai_params: Option<Box<[String]>>,
    /// `additional_drop_params`: params dropped before the call.
    #[serde_as(as = "Option<TextEntries>")]
    pub additional_drop_params: Option<Box<[String]>>,
}

impl CallSettings {
    fn of(deployment: &Deployment) -> Self {
        let params = deployment
            .raw_params
            .iter()
            .map(|(key, value)| (key.as_str(), value));
        Self::deserialize(MapDeserializer::<_, serde_json::Error>::new(params)).unwrap_or_default()
    }
}

static NOTHING_REQUESTED: BTreeSet<String> = BTreeSet::new();

/// The params the check filters on: those the request sends, while `litellm.drop_params` is off.
pub(super) fn requested(facts: &PreCallFacts) -> &BTreeSet<String> {
    if facts.drop_params == Some(false) {
        &facts.sent_params
    } else {
        &NOTHING_REQUESTED
    }
}

/// Which of `left` the stage drops: those lacking a requested param, unless every one lacks
/// it, and then only those that reject it.
pub(super) fn failing(
    left: &[&Deployment],
    requested: &BTreeSet<String>,
    facts: &PreCallFacts,
) -> Vec<bool> {
    let lacking: Vec<bool> = left
        .iter()
        .map(|deployment| lacks(deployment, requested, facts))
        .collect();
    if lacking.is_empty() || lacking.contains(&false) {
        return lacking;
    }
    left.iter()
        .map(|deployment| rejects(deployment, requested, facts))
        .collect()
}

/// The error for a group with nothing left, when every one of its deployments rejects a
/// requested param.
pub(super) fn unsupported(
    model: &str,
    pool: &[&Deployment],
    requested: &BTreeSet<String>,
    facts: &PreCallFacts,
) -> Option<Error> {
    let every_rejects = !pool.is_empty()
        && pool
            .iter()
            .all(|deployment| rejects(deployment, requested, facts));
    every_rejects.then(|| Error::UnsupportedParams {
        model: model.to_owned(),
        params: requested.iter().cloned().collect(),
    })
}

/// `_deployment_lacks_params`: it does not support one of `params`.
fn lacks<'a>(
    deployment: &Deployment,
    params: impl IntoIterator<Item = &'a String>,
    facts: &PreCallFacts,
) -> bool {
    let unsupported = &facts.of(deployment).unsupported_params;
    params.into_iter().any(|param| unsupported.contains(param))
}

/// `_deployment_rejects_params`: it lacks a requested param its call settings neither drop
/// nor allow.
fn rejects(deployment: &Deployment, requested: &BTreeSet<String>, facts: &PreCallFacts) -> bool {
    let own = CallSettings::of(deployment);
    let sent = &facts.call_settings;
    if sent.drop_params.or(own.drop_params) == Some(true) {
        return false;
    }
    let allowed = sent
        .allowed_openai_params
        .as_deref()
        .or(own.allowed_openai_params.as_deref())
        .unwrap_or_default();
    let dropped = sent
        .additional_drop_params
        .as_deref()
        .or(own.additional_drop_params.as_deref())
        .unwrap_or_default();
    let kept = requested
        .iter()
        .filter(|param| !allowed.contains(param) && !dropped.contains(param));
    lacks(deployment, kept, facts)
}
