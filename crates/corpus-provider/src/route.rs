//! The route table: which endpoint, which credential, which wire protocol.
//!
//! The same document the swarm worker reads. That is the point of writing it twice: a
//! provider added for the hosted deployment is a provider `corpus` can already address
//! from a laptop, with the same key names and the same model ids, so the two do not
//! drift into two catalogues that have to be kept in step by hand.
//!
//! A model is named `provider/model`. The prefix is split off only when it names a
//! configured route, because provider model ids carry slashes of their own —
//! `z-ai/glm-5.3` is one model, not a model called `glm-5.3` on a route called `z-ai`.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// The protocols a route may name. Deliberately the two the harness can describe with a
/// key, an endpoint and a set of headers; the rest need ambient credentials or a signing
/// scheme that configuration cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    #[default]
    OpenAiCompletions,
    AnthropicMessages,
}

impl Protocol {
    pub fn parse(name: &str) -> Result<Protocol> {
        match name {
            "openai-completions" => Ok(Protocol::OpenAiCompletions),
            "anthropic-messages" => Ok(Protocol::AnthropicMessages),
            other => bail!(
                "unknown protocol {other:?}; supported protocols are \
                 openai-completions, anthropic-messages"
            ),
        }
    }
}

/// What a route serves when neither the model nor the route names a ceiling. Anthropic
/// refuses a request without one, so a number always has to exist.
pub const DEFAULT_MAX_TOKENS: u32 = 32_768;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelDoc {
    id: String,
    name: Option<String>,
    context_window: Option<u32>,
    max_tokens: Option<u32>,
    /// Level a caller may select -> the spelling that goes on the wire. A null value
    /// means the level is offered and asking for it means sending nothing at all.
    reasoning_efforts: Option<BTreeMap<String, Option<String>>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RouteDoc {
    display_name: Option<String>,
    api: Option<String>,
    // Spelled out rather than left to the rename rule: camelCase turns `base_url` into
    // `baseUrl`, and the document this shares with the worker writes `baseURL`.
    #[serde(rename = "baseURL")]
    base_url: Option<String>,
    api_key_env: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    #[serde(default)]
    models: Vec<ModelDoc>,
}

#[derive(Debug, Clone, Deserialize)]
struct TableDoc {
    #[serde(default)]
    providers: BTreeMap<String, RouteDoc>,
}

/// One model a route serves, as configuration describes it.
#[derive(Debug, Clone)]
pub struct RouteModel {
    pub id: String,
    pub name: String,
    pub window: u32,
    pub max_tokens: Option<u32>,
    pub reasoning_efforts: BTreeMap<String, Option<String>>,
}

/// One provider this process can reach, with its credential already resolved.
#[derive(Debug, Clone)]
pub struct Route {
    pub provider: String,
    pub display_name: String,
    pub protocol: Protocol,
    pub base_url: String,
    pub api_key: String,
    pub headers: BTreeMap<String, String>,
    pub default_max_tokens: u32,
    pub models: Vec<RouteModel>,
}

impl Route {
    pub fn model(&self, id: &str) -> Option<&RouteModel> {
        self.models.iter().find(|entry| entry.id == id)
    }

    pub fn max_tokens_for(&self, id: &str) -> u32 {
        self.model(id)
            .and_then(|entry| entry.max_tokens)
            .unwrap_or(self.default_max_tokens)
    }

    /// How a selected reasoning level is spelled for this exact model: whether it is
    /// offered at all, and what to send. An unoffered level is refused rather than
    /// clamped — a request that silently did not think is worse than one that failed.
    pub fn effort_wire(&self, id: &str, level: &str) -> Option<Option<&str>> {
        let efforts = &self.model(id)?.reasoning_efforts;
        efforts.get(level).map(|spelling| spelling.as_deref())
    }
}

/// Every configured route, plus the endpoint this process was started with.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    routes: Vec<Route>,
    /// The route a name with no recognised prefix goes to. Always present in practice:
    /// it is `CORPUS_BASE_URL`, which has a default of its own.
    fallback: Option<String>,
}

impl Registry {
    /// The fallback alone, which is every deployment that configured no table.
    pub fn single(base_url: &str, api_key: &str) -> Registry {
        Registry {
            routes: vec![Route {
                provider: FALLBACK.to_string(),
                display_name: "Default".to_string(),
                protocol: Protocol::OpenAiCompletions,
                base_url: base_url.trim_end_matches('/').to_string(),
                api_key: api_key.to_string(),
                headers: BTreeMap::new(),
                default_max_tokens: DEFAULT_MAX_TOKENS,
                models: Vec::new(),
            }],
            fallback: Some(FALLBACK.to_string()),
        }
    }

    /// Parse a table and add the fallback under its own name, so a caller may address
    /// the process's own endpoint explicitly as well as by omission.
    pub fn parse(document: &str, base_url: &str, api_key: &str) -> Result<Registry> {
        let doc: TableDoc =
            serde_json::from_str(document).context("the route table is not valid JSON")?;
        let mut registry = Registry::single(base_url, api_key);
        for (provider, raw) in doc.providers {
            if provider.contains('/') {
                bail!(
                    "route {provider:?} may not contain a slash: the slash is what \
                     separates a route from the model it serves"
                );
            }
            let route = build(&provider, raw)?;
            // A table naming the fallback replaces it, which is how a bench points the
            // default somewhere else without every caller having to name a provider.
            match registry.routes.iter().position(|r| r.provider == provider) {
                Some(at) => registry.routes[at] = route,
                None => registry.routes.push(route),
            }
        }
        Ok(registry)
    }

    /// Split `provider/model` into the route serving it and the id it knows it by.
    pub fn resolve(&self, model: &str) -> Result<(&Route, String)> {
        let name = model.trim();
        if let Some((prefix, rest)) = name.split_once('/')
            && !rest.is_empty()
            && let Some(route) = self.routes.iter().find(|r| r.provider == prefix)
        {
            return Ok((route, rest.to_string()));
        }
        let fallback = self.fallback.as_deref().unwrap_or(FALLBACK);
        let route = self
            .routes
            .iter()
            .find(|r| r.provider == fallback)
            .with_context(|| {
                format!(
                    "no route serves {name:?} and there is no fallback route; \
                     configured routes are {}",
                    self.names().join(", ")
                )
            })?;
        Ok((route, name.to_string()))
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    pub fn names(&self) -> Vec<String> {
        self.routes.iter().map(|r| r.provider.clone()).collect()
    }

    pub fn route(&self, provider: &str) -> Option<&Route> {
        self.routes.iter().find(|r| r.provider == provider)
    }

    /// Whether a model id needs its route named to be unambiguous. The fallback's models
    /// are addressed bare, every other route's are qualified.
    pub fn qualify(&self, provider: &str, model: &str) -> String {
        match Some(provider) == self.fallback.as_deref() {
            true => model.to_string(),
            false => format!("{provider}/{model}"),
        }
    }
}

/// The name the process's own endpoint is registered under.
pub const FALLBACK: &str = "default";

fn build(provider: &str, raw: RouteDoc) -> Result<Route> {
    let protocol = match raw.api.as_deref() {
        Some(name) => Protocol::parse(name).with_context(|| format!("route {provider:?}"))?,
        None => Protocol::OpenAiCompletions,
    };
    let base_url = raw
        .base_url
        .filter(|url| !url.trim().is_empty())
        .with_context(|| format!("route {provider:?}: baseURL is required"))?;
    // A named-but-missing reference fails here rather than at request time: the request
    // would otherwise go out unauthenticated and come back as a provider 401, which reads
    // as a wrong key rather than as a key that was never set.
    let api_key = match raw.api_key_env.as_deref().filter(|name| !name.is_empty()) {
        Some(name) => std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .with_context(|| {
                format!("route {provider:?}: apiKeyEnv names {name}, which is unset or empty")
            })?,
        None => String::new(),
    };
    Ok(Route {
        provider: provider.to_string(),
        display_name: raw.display_name.unwrap_or_else(|| provider.to_string()),
        protocol,
        base_url: base_url.trim().trim_end_matches('/').to_string(),
        api_key,
        headers: raw.headers,
        default_max_tokens: raw
            .default_max_tokens
            .or(raw.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        models: raw
            .models
            .into_iter()
            .map(|entry| RouteModel {
                name: entry.name.clone().unwrap_or_else(|| entry.id.clone()),
                window: entry.context_window.unwrap_or(0),
                max_tokens: entry.max_tokens,
                reasoning_efforts: entry.reasoning_efforts.unwrap_or_default(),
                id: entry.id,
            })
            .collect(),
    })
}
