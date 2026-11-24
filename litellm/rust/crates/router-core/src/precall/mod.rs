//! `enable_pre_call_checks` (`_pre_call_checks`): before a pick, drop the deployments whose
//! context window the prompt exceeds, those outside the request's `allowed_model_region`, and
//! those that cannot take a requested param, then decide which error a group with nothing left
//! raises. Each stage is its own module. The facts that need live data (the prompt's token
//! count, each deployment's context window, which params it does not support, the live
//! `litellm.drop_params`) come from [`RouterHost::pre_call_facts`]; every decision is made here.
//!
//! [`RouterHost::pre_call_facts`]: crate::host::RouterHost::pre_call_facts

mod context_window;
mod params;
mod region;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use context_window::Overflow;
use litellm_python_compat::lenient::LaxBool;
use serde::Deserialize;
use serde_json::Value;
use serde_with::serde_as;

pub use params::CallSettings;

use crate::{error::Error, registry::Deployment};

/// What the host knows about one deployment for the pre-call checks.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct DeploymentFacts {
    /// The deployment's context window (`get_router_model_info`'s `max_input_tokens`), when it
    /// declares one and resolves.
    pub max_input_tokens: Option<i64>,
    /// The sent params it does not support (`_deployment_supports_param`).
    pub unsupported_params: BTreeSet<String>,
}

static NO_FACTS: DeploymentFacts = DeploymentFacts {
    max_input_tokens: None,
    unsupported_params: BTreeSet::new(),
};

/// What the host knows about a request for the pre-call checks.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PreCallFacts {
    /// The request carries `messages` or `input`: without either no check runs.
    pub has_input: bool,
    /// The prompt may be counted against a context window (no compaction is pending).
    pub countable: bool,
    /// The prompt's token count, when it was counted.
    pub input_tokens: Option<i64>,
    /// The request's `allowed_model_region`.
    pub allowed_model_region: Option<Value>,
    /// The live `litellm.drop_params`. Params are checked only while it is `false`.
    #[serde_as(as = "LaxBool")]
    pub drop_params: Option<bool>,
    /// The filtered params the request sends as anything but text
    /// (`_pre_call_check_requested_chat_params`).
    pub sent_params: BTreeSet<String>,
    /// The request's own call settings, which win over each deployment's.
    pub call_settings: CallSettings,
    /// Facts by deployment id, for the candidates and the group's deployments.
    pub deployments: BTreeMap<String, DeploymentFacts>,
}

impl PreCallFacts {
    fn of(&self, deployment: &Deployment) -> &DeploymentFacts {
        self.deployments.get(&deployment.id).unwrap_or(&NO_FACTS)
    }
}

/// `_pre_call_checks` over `candidates`; `pool` is the group's deployments before health filters
/// (`request_deployments`), which decide whether nothing left is an unsupported-params error.
pub fn filter(
    model: &str,
    candidates: Vec<Arc<Deployment>>,
    pool: &[Arc<Deployment>],
    facts: &PreCallFacts,
) -> Result<Vec<Arc<Deployment>>, Error> {
    if !facts.has_input {
        return Ok(candidates);
    }
    let overflows: Vec<Option<Overflow>> = candidates
        .iter()
        .map(|deployment| context_window::overflow(deployment, facts))
        .collect();
    let context_window_error = overflows.iter().any(Option::is_some);
    let dropped: Vec<bool> = candidates
        .iter()
        .zip(&overflows)
        .map(|(deployment, overflow)| overflow.is_some() || region::outside(deployment, facts))
        .collect();
    let requested = params::requested(facts);
    let left: Vec<&Deployment> = candidates
        .iter()
        .zip(&dropped)
        .filter(|(_, dropped)| !**dropped)
        .map(|(deployment, _)| &**deployment)
        .collect();
    let mut failing = params::failing(&left, requested, facts).into_iter();
    let kept: Vec<bool> = dropped
        .iter()
        .map(|dropped| !dropped && failing.next() == Some(false))
        .collect();
    if !kept.contains(&true) {
        let pool: Vec<&Deployment> = pool.iter().map(|deployment| &**deployment).collect();
        if !context_window_error
            && let Some(error) = params::unsupported(model, &pool, requested, facts)
        {
            return Err(error);
        }
        if context_window_error {
            return Err(context_window::exceeded(model, overflows.iter().flatten()));
        }
    }
    Ok(candidates
        .into_iter()
        .zip(kept)
        .filter_map(|(deployment, kept)| kept.then_some(deployment))
        .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    fn deployment(id: &str, params: Value, model_info: Value) -> Arc<Deployment> {
        let mut info = model_info;
        info["id"] = json!(id);
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": params, "model_info": info,
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn facts(entries: &[(&str, DeploymentFacts)]) -> PreCallFacts {
        PreCallFacts {
            has_input: true,
            countable: true,
            deployments: entries
                .iter()
                .map(|(id, f)| ((*id).to_owned(), f.clone()))
                .collect(),
            ..PreCallFacts::default()
        }
    }

    fn window(limit: i64) -> DeploymentFacts {
        DeploymentFacts {
            max_input_tokens: Some(limit),
            ..DeploymentFacts::default()
        }
    }

    #[rstest]
    fn a_prompt_over_a_context_window_drops_that_deployment() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let big = deployment("big", json!({"model": "openai/b"}), json!({}));
        let mut found = facts(&[("small", window(5)), ("big", window(10_000))]);
        found.input_tokens = Some(100);
        let kept = filter("g", vec![small, big], &[], &found).unwrap();
        assert_eq!(
            kept.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["big"]
        );
    }

    #[rstest]
    fn no_window_left_names_each_deployment_back_to_back() {
        let one = deployment("one", json!({"model": "openai/one"}), json!({}));
        let two = deployment(
            "two",
            json!({"model": "openai/two"}),
            json!({"base_model": "azure/base"}),
        );
        let mut found = facts(&[("one", window(5)), ("two", window(6))]);
        found.input_tokens = Some(9);
        let error = filter("g", vec![one, two], &[], &found).unwrap_err();
        let Error::ContextWindowExceeded { model, detail } = error else {
            panic!("expected a context window error, got {error:?}");
        };
        assert_eq!(model, "g");
        assert_eq!(
            detail,
            "Model=openai/one, Max Input Tokens=5, Got=9Model=azure/base, Max Input Tokens=6, Got=9"
        );
    }

    #[rstest]
    fn without_input_nothing_is_checked() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", window(5))]);
        found.input_tokens = Some(100);
        found.has_input = false;
        assert_eq!(filter("g", vec![small], &[], &found).unwrap().len(), 1);
    }

    #[rstest]
    fn a_prompt_that_may_not_be_counted_fits_every_window() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", window(5))]);
        found.input_tokens = Some(100);
        found.countable = false;
        assert_eq!(filter("g", vec![small], &[], &found).unwrap().len(), 1);
    }

    #[rstest]
    #[case::in_region(json!("eu"), &["eu"])]
    #[case::out_of_region(json!("ap"), &[])]
    #[case::null_region(Value::Null, &["eu", "none"])]
    fn the_allowed_region_keeps_only_its_deployments(#[case] region: Value, #[case] kept: &[&str]) {
        let eu = deployment(
            "eu",
            json!({"model": "openai/a", "region_name": "eu"}),
            json!({}),
        );
        let none = deployment("none", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[]);
        found.allowed_model_region = Some(region);
        let left = filter("g", vec![eu, none], &[], &found).unwrap();
        assert_eq!(left.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(), kept);
    }

    fn lacking() -> DeploymentFacts {
        DeploymentFacts {
            unsupported_params: BTreeSet::from(["response_format".to_owned()]),
            ..DeploymentFacts::default()
        }
    }

    fn sending_response_format(entries: &[(&str, DeploymentFacts)]) -> PreCallFacts {
        PreCallFacts {
            drop_params: Some(false),
            sent_params: BTreeSet::from(["response_format".to_owned()]),
            ..facts(entries)
        }
    }

    fn ids(kept: &[Arc<Deployment>]) -> Vec<&str> {
        kept.iter().map(|d| d.id.as_str()).collect()
    }

    #[rstest]
    fn params_are_checked_only_on_deployments_inside_the_region() {
        let eu = deployment(
            "eu",
            json!({"model": "openai/old", "region_name": "eu", "drop_params": true}),
            json!({}),
        );
        let ap = deployment(
            "ap",
            json!({"model": "openai/new", "region_name": "ap"}),
            json!({}),
        );
        let mut found = sending_response_format(&[("eu", lacking())]);
        found.allowed_model_region = Some(json!("eu"));
        let kept = filter("g", vec![eu, ap], &[], &found).unwrap();
        assert_eq!(ids(&kept), ["eu"]);
    }

    #[rstest]
    fn a_group_where_every_deployment_rejects_the_param_is_unsupported() {
        let old = deployment("old", json!({"model": "openai/old"}), json!({}));
        let found = sending_response_format(&[("old", lacking())]);
        let error = filter("g", vec![old.clone()], &[old], &found).unwrap_err();
        assert!(
            matches!(error, Error::UnsupportedParams { params, .. } if params == ["response_format"])
        );
    }

    #[rstest]
    fn a_deployment_that_takes_the_param_wins_over_one_that_would_drop_it() {
        let old = deployment(
            "old",
            json!({"model": "openai/old", "drop_params": true}),
            json!({}),
        );
        let new = deployment("new", json!({"model": "openai/new"}), json!({}));
        let found = sending_response_format(&[("old", lacking())]);
        let kept = filter("g", vec![old, new], &[], &found).unwrap();
        assert_eq!(ids(&kept), ["new"]);
    }

    #[rstest]
    #[case::rejected(json!({}), json!({}), Some(false), 0)]
    #[case::own_drop_params(json!({"drop_params": "true"}), json!({}), Some(false), 1)]
    #[case::own_allowlist(json!({"allowed_openai_params": ["response_format"]}), json!({}), Some(false), 1)]
    #[case::own_drop_list(json!({"additional_drop_params": ["response_format"]}), json!({}), Some(false), 1)]
    #[case::sent_allowlist_text(json!({}), json!({"allowed_openai_params": "tools,response_format"}), Some(false), 1)]
    #[case::sent_drop_params_wins(json!({"drop_params": true}), json!({"drop_params": false}), Some(false), 0)]
    #[case::sent_allowlist_wins(
        json!({"allowed_openai_params": ["response_format"]}),
        json!({"allowed_openai_params": ["tools"]}),
        Some(false),
        0
    )]
    #[case::global_drop_params(json!({}), json!({}), Some(true), 1)]
    #[case::global_drop_params_unset(json!({}), json!({}), None, 1)]
    fn when_every_deployment_lacks_the_param_only_those_rejecting_it_are_dropped(
        #[case] own: Value,
        #[case] sent: Value,
        #[case] global_drop_params: Option<bool>,
        #[case] kept: usize,
    ) {
        let mut params = own;
        params["model"] = json!("openai/old");
        let old = deployment("old", params, json!({}));
        let found = PreCallFacts {
            drop_params: global_drop_params,
            call_settings: serde_json::from_value(sent).unwrap(),
            ..sending_response_format(&[("old", lacking())])
        };
        assert_eq!(filter("g", vec![old], &[], &found).unwrap().len(), kept);
    }
}
