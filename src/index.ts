import type { Plugin, Config } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import { homedir } from "os";
import { join } from "path";
import { readFile } from "fs/promises";
import { platform } from "process";
import { createRequire } from "module";

interface ApertureModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  metadata?: {
    provider?: {
      id: string;
      name: string;
      description?: string;
    };
  };
}

interface ApertureResponse {
  object: string;
  data?: ApertureModel[];
  models?: Array<ApertureModel & {
    model?: string;
  }>;
}

type ApertureProviderCompatibility = {
  openai_chat?: boolean;
  openai_responses?: boolean;
  anthropic_messages?: boolean;
};

type ApertureProviderMetadata = {
  id: string;
  name?: string;
  description?: string;
  models?: string[];
  compatibility?: ApertureProviderCompatibility;
};

type ApertureProviderFetchResult = {
  providers: Map<string, ApertureProviderMetadata>;
  degraded: boolean;
};

interface ApertureConfig {
  baseUrl?: string;
  apiKey?: string;
  modelsDevUrl?: string;
  modelsDevPath?: string;
  disableModelsDev?: boolean;
}

type InterleavedConfig = true | {
  field: "reasoning_content" | "reasoning_details";
};

type ModelCost = {
  input: number;
  output: number;
  cache_read?: number;
  cache_write?: number;
  context_over_200k?: {
    input: number;
    output: number;
    cache_read?: number;
    cache_write?: number;
  };
};

type ApertureModelConfig = {
  id?: string;
  name?: string;
  family?: string;
  release_date?: string;
  attachment?: boolean;
  status?: "alpha" | "beta" | "deprecated";
  limit?: {
    context: number;
    input?: number;
    output: number;
  };
  cost?: ModelCost;
  reasoning?: boolean;
  temperature?: boolean;
  tool_call?: boolean;
  modalities?: {
    input: Array<"text" | "audio" | "image" | "video" | "pdf">;
    output: Array<"text" | "audio" | "image" | "video" | "pdf">;
  };
  interleaved?: InterleavedConfig;
  options?: {
    thinking?: {
      type?: string;
    };
    [key: string]: unknown;
  };
  headers?: Record<string, string>;
  variants?: Record<string, Record<string, unknown>>;
};

type ThinkingConfig = {
  type?: string;
};

type ToastVariant = "success" | "error";

type PendingToast = {
  variant: ToastVariant;
  message: string;
  attempts: number;
};

type ModelsDevModel = {
  id: string;
  name: string;
  family?: string;
  release_date?: string;
  attachment?: boolean;
  status?: "alpha" | "beta" | "deprecated";
  cost?: ModelCost;
  limit?: {
    context: number;
    input?: number;
    output: number;
  };
  reasoning?: boolean;
  temperature?: boolean;
  tool_call?: boolean;
  modalities?: {
    input: Array<"text" | "audio" | "image" | "video" | "pdf">;
    output: Array<"text" | "audio" | "image" | "video" | "pdf">;
  };
  interleaved?: InterleavedConfig;
  reasoning_options?: Array<{
    type: "effort" | "toggle" | "budget_tokens" | string;
    values?: string[];
    min?: number;
    max?: number;
  }>;
};

type ModelsDevProvider = {
  id: string;
  name: string;
  npm?: string;
  api?: string;
  models: Record<string, ModelsDevModel>;
};

type ModelsDevCatalog = Record<string, ModelsDevProvider>;

type ApertureWireAPI = "openai" | "anthropic";

type ModelDefaultsResult = {
  defaults: Omit<ApertureModelConfig, "id" | "name">;
  matchedModelsDev: boolean;
};

const STARTUP_MODEL_STABILIZATION_DEADLINE_MS = 4_000;
const STARTUP_FETCH_TIMEOUT_MS = 2_000;
const STARTUP_POLL_INTERVAL_MS = 250;
const STARTUP_MIN_FETCH_TIMEOUT_MS = 750;
const INTERACTIVE_FETCH_TIMEOUT_MS = 5_000;
const MODELS_DEV_FETCH_TIMEOUT_MS = 3_000;

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

type ApertureProviderGroup = {
  id: string;
  name: string;
  routeProviderID?: string;
  wireAPI: ApertureWireAPI;
  compatibility?: ApertureProviderCompatibility;
};

function slugifyProviderSegment(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return normalized || "default";
}

function getProviderDisplayName(providerID: string): string {
  const slug = slugifyProviderSegment(providerID);
  return providerID.trim() || slug;
}

function normalizeModelLookup(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[_:\s.]+/g, "-")
    .replace(/\/+/g, "/")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function findProviderModel(provider: ModelsDevProvider, modelKeys: Set<string>): ModelsDevModel | undefined {
  for (const key of modelKeys) {
    const candidate = provider.models[key];
    if (candidate) {
      return candidate;
    }
  }

  return undefined;
}

function findModelsDevEntry(
  model: ApertureModel,
  catalog?: ModelsDevCatalog,
  apertureProvider?: ApertureProviderMetadata,
): {
  provider: ModelsDevProvider;
  model: ModelsDevModel;
} | undefined {
  if (!catalog) {
    return undefined;
  }

  const modelKeys = new Set([
    model.id,
    model.id.toLowerCase(),
    normalizeModelLookup(model.id),
  ]);
  const apertureProviderID = apertureProvider?.id ?? model.metadata?.provider?.id;
  const provider = apertureProviderID ? catalog[apertureProviderID] ?? catalog[apertureProviderID.toLowerCase()] : undefined;

  if (provider) {
    const candidate = findProviderModel(provider, modelKeys);
    if (candidate) {
      return { provider, model: candidate };
    }
  }

  const exactMatches: Array<{ provider: ModelsDevProvider; model: ModelsDevModel }> = [];
  for (const provider of Object.values(catalog)) {
    for (const key of modelKeys) {
      const candidate = provider.models[key];
      if (candidate) {
        exactMatches.push({ provider, model: candidate });
      }
    }
  }

  return exactMatches.length === 1 ? exactMatches[0] : undefined;
}

function getProviderWireAPI(provider?: ApertureProviderMetadata): ApertureWireAPI {
  const compatibility = provider?.compatibility;
  if (compatibility?.openai_chat || compatibility?.openai_responses) {
    return "openai";
  }
  if (compatibility?.anthropic_messages) {
    return "anthropic";
  }
  return "openai";
}

function getProviderGroup(model: ApertureModel, providers?: Map<string, ApertureProviderMetadata>): ApertureProviderGroup {
  const providerID = model.metadata?.provider?.id?.trim();
  const providerName = model.metadata?.provider?.name?.trim();
  const providerSegment = providerName || providerID;
  const routeProviderID = providerID || providerName;
  const displayName = providerName || (providerSegment ? getProviderDisplayName(providerSegment) : undefined);
  const providerMetadata = routeProviderID ? providers?.get(routeProviderID) : undefined;
  const wireAPI = getProviderWireAPI(providerMetadata);

  if (!providerSegment || !displayName) {
    return {
      id: "aperture",
      name: "Aperture",
      wireAPI,
      compatibility: providerMetadata?.compatibility,
    };
  }

  return {
    id: `aperture-${slugifyProviderSegment(providerSegment)}`,
    name: `Aperture/${displayName}`,
    routeProviderID,
    wireAPI,
    compatibility: providerMetadata?.compatibility,
  };
}

function getModelProviderKey(model: ApertureModel, providers?: Map<string, ApertureProviderMetadata>): string {
  const group = getProviderGroup(model, providers);
  return `${group.id}:${group.wireAPI}:${model.id}`;
}

function getApertureRouteModelID(model: ApertureModel, providers?: Map<string, ApertureProviderMetadata>): string {
  const routeProviderID = getProviderGroup(model, providers).routeProviderID;
  return routeProviderID ? `${routeProviderID}/${model.id}` : model.id;
}

const PROTOCOL_NPM: Array<[string, string]> = [
  ["openai_responses", "@ai-sdk/openai"],
  ["openai_chat", "@ai-sdk/openai-compatible"],
  ["anthropic_messages", "@ai-sdk/anthropic"],
  ["gemini_generate_content", "@ai-sdk/google"],
  ["google_generate_content", "@ai-sdk/google"],
  ["google_raw_predict", "@ai-sdk/google"],
  ["bedrock_converse", "@ai-sdk/amazon-bedrock"],
  ["bedrock_model_invoke", "@ai-sdk/amazon-bedrock"],
];

function getProviderNpmPackage(wireAPI: ApertureWireAPI, compatibility?: ApertureProviderCompatibility): string {
  const compat = compatibility as Record<string, boolean | undefined> | undefined;
  for (const [key, pkg] of PROTOCOL_NPM) {
    if (compat?.[key]) return pkg;
  }
  return wireAPI === "anthropic" ? "@ai-sdk/anthropic" : "@ai-sdk/openai-compatible";
}

function getCatalogReasoningVariants(model: ModelsDevModel): Record<string, Record<string, unknown>> | undefined {
  const effort = model.reasoning_options?.find((option) => option.type === "effort");
  const values = effort?.values
    ?.map((value) => value.trim())
    .filter((value) => value.length > 0);

  if (!values || values.length === 0) {
    return undefined;
  }

  return Object.fromEntries(values.map((value) => [
    value,
    { reasoningEffort: value },
  ]));
}

function getModelsDevDefaults(entry: {
  provider: ModelsDevProvider;
  model: ModelsDevModel;
}): Omit<ApertureModelConfig, "id" | "name"> {
  const defaults: Omit<ApertureModelConfig, "id" | "name"> = {
    family: entry.model.family,
    release_date: entry.model.release_date,
    attachment: entry.model.attachment,
    status: entry.model.status,
    cost: entry.model.cost,
    limit: entry.model.limit,
    reasoning: entry.model.reasoning,
    temperature: entry.model.temperature,
    tool_call: entry.model.tool_call,
    modalities: entry.model.modalities,
    interleaved: entry.model.interleaved,
  };

  const variants = getCatalogReasoningVariants(entry.model);
  if (variants && Object.keys(variants).length > 0) {
    defaults.variants = variants;
  }

  return Object.fromEntries(
    Object.entries(defaults).filter(([, value]) => value !== undefined),
  ) as Omit<ApertureModelConfig, "id" | "name">;
}

function getModelDefaults(
  model: ApertureModel,
  catalog?: ModelsDevCatalog,
  providers?: Map<string, ApertureProviderMetadata>,
): ModelDefaultsResult {
  const routeProviderID = getProviderGroup(model, providers).routeProviderID;
  const apertureProvider = routeProviderID ? providers?.get(routeProviderID) : undefined;
  const modelsDevEntry = findModelsDevEntry(model, catalog, apertureProvider);
  if (modelsDevEntry) {
    return {
      defaults: getModelsDevDefaults(modelsDevEntry),
      matchedModelsDev: true,
    };
  }

  return {
    defaults: {
      limit: {
        context: 128_000,
        output: 8_192,
      },
      reasoning: false,
      temperature: true,
      tool_call: true,
      modalities: {
        input: ["text"],
        output: ["text"],
      },
      interleaved: {
        field: "reasoning_content",
      },
    },
    matchedModelsDev: false,
  };
}

function getDefaultReleaseDate(created?: number): string {
  if (!created || created <= 0) {
    return "";
  }

  return new Date(created * 1000).toISOString().slice(0, 10);
}

function mergeThinkingConfig(defaults?: ThinkingConfig, existing?: ThinkingConfig): ThinkingConfig | undefined {
  if (!defaults && !existing) {
    return undefined;
  }

  return {
    ...defaults,
    ...existing,
  };
}

function mergeModelConfig(defaults: Omit<ApertureModelConfig, "id" | "name">, existing: ApertureModelConfig = {}): ApertureModelConfig {
  const thinking = mergeThinkingConfig(
    defaults.options?.thinking as ThinkingConfig | undefined,
    existing.options?.thinking as ThinkingConfig | undefined,
  );
  const limit = defaults.limit || existing.limit ? {
    context: existing.limit?.context ?? defaults.limit?.context ?? 0,
    input: existing.limit?.input ?? defaults.limit?.input,
    output: existing.limit?.output ?? defaults.limit?.output ?? 0,
  } : undefined;
  const modalities = defaults.modalities || existing.modalities ? {
    input: existing.modalities?.input ?? defaults.modalities?.input ?? ["text"],
    output: existing.modalities?.output ?? defaults.modalities?.output ?? ["text"],
  } : undefined;
  const cost = defaults.cost || existing.cost ? {
    ...defaults.cost,
    ...existing.cost,
    ...(defaults.cost?.context_over_200k || existing.cost?.context_over_200k ? {
      context_over_200k: {
        ...defaults.cost?.context_over_200k,
        ...existing.cost?.context_over_200k,
      },
    } : {}),
  } as ModelCost : undefined;

  return {
    ...defaults,
    ...existing,
    ...(limit ? { limit } : {}),
    ...(cost ? { cost } : {}),
    ...(modalities ? { modalities } : {}),
    ...(defaults.interleaved || existing.interleaved ? {
      interleaved: existing.interleaved ?? defaults.interleaved,
    } : {}),
    ...(defaults.options || existing.options ? {
      options: {
        ...defaults.options,
        ...existing.options,
        ...(thinking ? { thinking } : {}),
      },
    } : {}),
    ...(defaults.headers || existing.headers ? {
      headers: {
        ...defaults.headers,
        ...existing.headers,
      },
    } : {}),
    ...(defaults.variants || existing.variants ? {
      variants: {
        ...defaults.variants,
        ...existing.variants,
      },
    } : {}),
  };
}

/**
 * Poll the models endpoint until the set of model IDs stabilizes (two
 * consecutive fetches return the same IDs) or the deadline is exceeded.
 * Transient fetch errors are retried within the deadline.
 */
async function waitForStableModels(
  baseUrl: string,
  apiKey: string,
  logger: Logger,
  {
    pollIntervalMs = STARTUP_POLL_INTERVAL_MS,
    deadlineMs = STARTUP_MODEL_STABILIZATION_DEADLINE_MS,
    fetchTimeoutMs = STARTUP_FETCH_TIMEOUT_MS,
    minFetchTimeoutMs = STARTUP_MIN_FETCH_TIMEOUT_MS,
    previousModels = [] as ApertureModel[],
    previousProviders = new Map<string, ApertureProviderMetadata>(),
  } = {},
): Promise<{ models: ApertureModel[]; providers: Map<string, ApertureProviderMetadata>; providersDegraded: boolean }> {
  const deadline = Date.now() + deadlineMs;
  let previousIds: string | undefined = previousModels.length > 0
    ? previousModels.map((model) => getModelProviderKey(model, previousProviders)).sort().join("\n")
    : undefined;
  let lastGoodResult: ApertureModel[] = previousModels;
  let lastGoodProviders = previousProviders;
  let lastGoodProvidersDegraded = false;
  let sawSuccessfulFetch = previousModels.length > 0;
  let lastError: unknown;

  while (Date.now() < deadline) {
    const remaining = deadline - Date.now();
    if (remaining < minFetchTimeoutMs && lastGoodResult.length > 0) {
      return { models: lastGoodResult, providers: lastGoodProviders, providersDegraded: lastGoodProvidersDegraded };
    }

    try {
      const providerResult = await fetchApertureProviders(baseUrl, apiKey, logger, Math.min(remaining, fetchTimeoutMs));
      const providers = providerResult.providers;
      const models = await fetchApertureModels(baseUrl, apiKey, logger, Math.min(remaining, fetchTimeoutMs), providers);
      const ids = models.map((model) => getModelProviderKey(model, providers)).sort().join("\n");

      lastGoodResult = models;
      lastGoodProviders = providers;
      lastGoodProvidersDegraded = providerResult.degraded;
      sawSuccessfulFetch = true;

      if (ids === previousIds) {
        return { models, providers, providersDegraded: providerResult.degraded };
      }
      previousIds = ids;
    } catch (error) {
      lastError = error;
      // Transient error — retry until deadline.
    }

    if (Date.now() + pollIntervalMs >= deadline) {
      return { models: lastGoodResult, providers: lastGoodProviders, providersDegraded: lastGoodProvidersDegraded };
    }

    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }

  if (!sawSuccessfulFetch && lastError) {
    throw lastError;
  }

  return { models: lastGoodResult, providers: lastGoodProviders, providersDegraded: lastGoodProvidersDegraded };
}

async function fetchApertureProviders(baseUrl: string, apiKey: string, logger: Logger, timeoutMs = INTERACTIVE_FETCH_TIMEOUT_MS): Promise<ApertureProviderFetchResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const url = `${baseUrl}/api/providers`;
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: apiKey ? {
        Authorization: `Bearer ${apiKey}`,
      } : undefined,
    });
    if (!response.ok) {
      logger.warn(`[TailscaleAperture] Aperture API request failed: GET /api/providers ${response.status} ${response.statusText}`);
      return { providers: new Map(), degraded: true };
    }

    const providers = await response.json() as ApertureProviderMetadata[];
    return { providers: new Map(providers.map((provider) => [provider.id, provider])), degraded: false };
  } catch (error) {
    logger.warn("[TailscaleAperture] Aperture API request failed: GET /api/providers", error);
    return { providers: new Map(), degraded: true };
  } finally {
    clearTimeout(timer);
  }
}

async function fetchApertureModels(
  baseUrl: string,
  apiKey: string,
  logger: Logger,
  timeoutMs = INTERACTIVE_FETCH_TIMEOUT_MS,
  providers = new Map<string, ApertureProviderMetadata>(),
): Promise<ApertureModel[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const url = `${baseUrl}/v1/models`;
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: apiKey ? {
        Authorization: `Bearer ${apiKey}`,
      } : undefined,
    });
    if (!response.ok) {
      logger.warn(`[TailscaleAperture] Aperture API request failed: GET /v1/models ${response.status} ${response.statusText}`);
      throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as ApertureResponse;
    const openAIModels = data.data ?? [];
    const llamaCppModels = (data.models ?? []).map((model) => ({
      ...model,
      id: model.id || model.model || "",
      object: model.object || "model",
      created: model.created || 0,
      owned_by: model.owned_by || "unknown",
    }));
    const mergedModels = [...openAIModels, ...llamaCppModels]
      .filter((model) => model.id);

    return Array.from(
      new Map(mergedModels.map((model) => [getModelProviderKey(model, providers), model])).values(),
    );
  } catch (error) {
    if (!(error instanceof Error && error.message.startsWith("Failed to fetch models:"))) {
      logger.warn("[TailscaleAperture] Aperture API request failed: GET /v1/models", error);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function readModelsDevCatalog(path: string, logger: Logger): Promise<ModelsDevCatalog | undefined> {
  try {
    const content = await readFile(path, "utf-8");
    return JSON.parse(content) as ModelsDevCatalog;
  } catch (error) {
    logger.warn(`[TailscaleAperture] Failed to read Models.dev catalog from ${path}:`, error);
    return undefined;
  }
}

async function fetchModelsDevCatalog(url: string, logger: Logger, timeoutMs = MODELS_DEV_FETCH_TIMEOUT_MS): Promise<ModelsDevCatalog | undefined> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const baseUrl = url.replace(/\/+$/, "");

  try {
    const response = await fetch(`${baseUrl}/api.json`, {
      signal: controller.signal,
      headers: {
        "User-Agent": "opencode-plugin-tsaperture",
      },
    });
    if (!response.ok) {
      logger.warn(`[TailscaleAperture] Models.dev request failed: GET /api.json ${response.status} ${response.statusText}`);
      return undefined;
    }

    return await response.json() as ModelsDevCatalog;
  } catch (error) {
    logger.warn("[TailscaleAperture] Models.dev request failed: GET /api.json", error);
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

async function loadModelsDevCatalog(config: ApertureConfig, logger: Logger): Promise<ModelsDevCatalog | undefined> {
  if (config.disableModelsDev || process.env.OPENCODE_DISABLE_MODELS_FETCH) {
    logger.info("[TailscaleAperture] Models.dev enrichment disabled");
    return undefined;
  }

  const path = config.modelsDevPath || process.env.OPENCODE_MODELS_PATH;
  if (path) {
    return readModelsDevCatalog(path, logger);
  }

  const url = config.modelsDevUrl || process.env.OPENCODE_MODELS_URL || "https://models.dev";
  return fetchModelsDevCatalog(url, logger);
}

function getOpenCodeConfigDirs(): string[] {
  const home = homedir();
  const dirs: string[] = [];

  if (platform === "win32") {
    dirs.push(join(process.env.APPDATA || process.env.LOCALAPPDATA || home, "opencode"));
  } else if (platform === "darwin") {
    const xdgConfig = process.env.XDG_CONFIG_HOME;
    if (xdgConfig) {
      dirs.push(join(xdgConfig, "opencode"));
    }
    dirs.push(join(home, ".config", "opencode"));
    dirs.push(join(home, "Library", "Application Support", "opencode"));
  } else {
    const xdgConfig = process.env.XDG_CONFIG_HOME;
    if (xdgConfig) {
      dirs.push(join(xdgConfig, "opencode"));
    }
    dirs.push(join(home, ".config", "opencode"));
  }

  return dirs;
}

type Logger = {
  log: (message: string, ...args: unknown[]) => void;
  warn: (message: string, ...args: unknown[]) => void;
  error: (message: string, ...args: unknown[]) => void;
  info: (message: string, ...args: unknown[]) => void;
  debug: (message: string, ...args: unknown[]) => void;
};

async function loadApertureConfig(logger: Logger): Promise<ApertureConfig> {
  for (const configDir of getOpenCodeConfigDirs()) {
    const configPath = join(configDir, "aperture.json");
    try {
      const content = await readFile(configPath, "utf-8");
      logger.log(`[TailscaleAperture] Loaded config from ${configPath}`);
      return JSON.parse(content) as ApertureConfig;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn(`[TailscaleAperture] Failed to read ${configPath}:`, error);
      }
    }
  }

  return {};
}

export const TailscaleAperturePlugin: Plugin = async (input, options) => {
  const client = input.client;

  const logger: Logger = {
    log: (message: string, ...args: unknown[]) => {
      client.app.log({
        body: {
          service: "TailscaleAperture",
          level: "info",
          message,
          extra: args.length > 0 ? { args: args.map((a) => a instanceof Error ? a.stack || a.message : String(a)) } : undefined,
        },
      }).catch(() => {});
    },
    warn: (message: string, ...args: unknown[]) => {
      client.app.log({
        body: {
          service: "TailscaleAperture",
          level: "warn",
          message,
          extra: args.length > 0 ? { args: args.map((a) => a instanceof Error ? a.stack || a.message : String(a)) } : undefined,
        },
      }).catch(() => {});
    },
    error: (message: string, ...args: unknown[]) => {
      client.app.log({
        body: {
          service: "TailscaleAperture",
          level: "error",
          message,
          extra: args.length > 0 ? { args: args.map((a) => a instanceof Error ? a.stack || a.message : String(a)) } : undefined,
        },
      }).catch(() => {});
    },
    info: (message: string, ...args: unknown[]) => {
      client.app.log({
        body: {
          service: "TailscaleAperture",
          level: "info",
          message,
          extra: args.length > 0 ? { args: args.map((a) => a instanceof Error ? a.stack || a.message : String(a)) } : undefined,
        },
      }).catch(() => {});
    },
    debug: (message: string, ...args: unknown[]) => {
      client.app.log({
        body: {
          service: "TailscaleAperture",
          level: "debug",
          message,
          extra: args.length > 0 ? { args: args.map((a) => a instanceof Error ? a.stack || a.message : String(a)) } : undefined,
        },
      }).catch(() => {});
    },
  };

  const require = createRequire(import.meta.url);
  const pkg = require("../package.json") as { name: string; version: string };
  logger.log(`[TailscaleAperture] ${pkg.name} v${pkg.version}`);

  const pendingToasts: PendingToast[] = [];
  let tuiReady = false;
  let toastFlushTimer: ReturnType<typeof setTimeout> | undefined;

  async function sendToast(toast: PendingToast): Promise<void> {
    const result = await client.tui.showToast({
      body: {
        title: "Tailscale Aperture",
        message: toast.message,
        variant: toast.variant,
        duration: 10_000,
      },
      query: {
        directory: input.directory,
      },
    });
    if (result.error) {
      throw new Error(`Failed to show opencode toast: ${JSON.stringify(result.error)}`);
    }
  }

  function scheduleToastFlush(): void {
    if (!tuiReady || toastFlushTimer) {
      return;
    }

    toastFlushTimer = setTimeout(() => {
      toastFlushTimer = undefined;
      flushToastQueue().catch((error) => {
        logger.warn("[TailscaleAperture] Failed to flush queued toasts:", error);
      });
    }, 1_000);
    toastFlushTimer.unref?.();
  }

  async function flushToastQueue(): Promise<void> {
    if (!tuiReady || pendingToasts.length === 0) {
      return;
    }

    const toasts = pendingToasts.splice(0, pendingToasts.length);
    for (const toast of toasts) {
      try {
        await sendToast(toast);
      } catch (error) {
        logger.warn("[TailscaleAperture] Failed to show opencode toast:", error);
        if (toast.attempts < 5) {
          pendingToasts.push({
            ...toast,
            attempts: toast.attempts + 1,
          });
        }
      }
    }

    if (pendingToasts.length > 0) {
      scheduleToastFlush();
    }
  }

  function showMessage(variant: ToastVariant, message: string): void {
    if (pendingToasts.some((toast) => toast.variant === variant && toast.message === message)) {
      return;
    }

    pendingToasts.push({
      variant,
      message,
      attempts: 0,
    });

    if (tuiReady) {
      flushToastQueue().catch((error) => {
        logger.warn("[TailscaleAperture] Failed to flush queued toasts:", error);
      });
    }
  }

  function markTuiReady(): void {
    tuiReady = true;
    flushToastQueue().catch((error) => {
      logger.warn("[TailscaleAperture] Failed to flush queued toasts:", error);
    });
  }

  const fileConfig = await loadApertureConfig(logger);
  const rawBaseUrl = (options?.baseUrl as string) || process.env.APERTURE_BASE_URL || fileConfig.baseUrl;
  const apiKey = (options?.apiKey as string) || process.env.APERTURE_API_KEY || fileConfig.apiKey || "";
  const modelsDevConfig: ApertureConfig = {
    ...fileConfig,
    modelsDevUrl: (options?.modelsDevUrl as string | undefined) ?? fileConfig.modelsDevUrl,
    modelsDevPath: (options?.modelsDevPath as string | undefined) ?? fileConfig.modelsDevPath,
    disableModelsDev: (options?.disableModelsDev as boolean | undefined) ?? fileConfig.disableModelsDev,
  };

  if (!rawBaseUrl) {
    const message = "No baseUrl configured. Set APERTURE_BASE_URL, add baseUrl to plugin options, or create aperture.json in opencode config directory.";
    logger.warn(`[TailscaleAperture] ${message}`);
    showMessage("error", message);
    return {
      config: async () => {
        markTuiReady();
      },
      event: async ({ event }) => {
        if (event.type === "server.connected") {
          markTuiReady();
        }
      },
    };
  }

  if (!apiKey) {
    logger.info("[TailscaleAperture] No API key configured. This may be okay if you don't use authorization.");
  }

  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  let discoveredModels: ApertureModel[] = [];
  let discoveredProviders = new Map<string, ApertureProviderMetadata>();
  let providerMetadataDegraded = false;
  let providerMetadataWarningShown = false;
  let modelsDevCatalog: ModelsDevCatalog | undefined;
  let modelsLoaded = false;
  let modelLoadPromise: Promise<ApertureModel[]> | undefined;
  const warnedModelsDevFallbacks = new Set<string>();

  function formatError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function formatDuration(ms: number | undefined): string {
    return ms === undefined ? "unknown" : `${ms}ms`;
  }

  async function printErrorToChat(message: string): Promise<void> {
    try {
      const sessionsResult = await client.session.list({
        query: {
          directory: input.directory,
        },
      });
      if (sessionsResult.error) {
        throw new Error(`Failed to list opencode sessions: ${JSON.stringify(sessionsResult.error)}`);
      }

      const session = sessionsResult.data
        ?.filter((candidate) => candidate.directory === input.directory)
        .sort((a, b) => b.time.updated - a.time.updated)[0];
      if (!session) {
        logger.warn("[TailscaleAperture] Failed to print error to chat: no opencode session found");
        return;
      }

      const promptResult = await client.session.promptAsync({
        path: {
          id: session.id,
        },
        query: {
          directory: input.directory,
        },
        body: {
          noReply: true,
          parts: [{
            type: "text",
            text: message,
            synthetic: true,
          }],
        },
      });
      if (promptResult.error) {
        throw new Error(`Failed to print error to chat: ${JSON.stringify(promptResult.error)}`);
      }
    } catch (error) {
      logger.warn("[TailscaleAperture] Failed to print error to chat:", error);
    }
  }

  function warnProviderMetadataDegraded(): void {
    if (!providerMetadataDegraded || providerMetadataWarningShown) {
      return;
    }

    providerMetadataWarningShown = true;
    const message = "Aperture provider metadata could not be loaded. Models were registered in degraded mode; provider grouping or wire API selection may be less accurate.";
    logger.warn(`[TailscaleAperture] ${message}`);
    showMessage("error", message);
  }

  async function loadModels(refresh = false): Promise<ApertureModel[]> {
    if (!refresh && modelsLoaded) {
      return discoveredModels;
    }

    if (refresh && modelsLoaded) {
      // Interactive refresh: single fetch, no stabilization wait.
      const providerResult = await fetchApertureProviders(baseUrl, apiKey, logger);
      discoveredProviders = providerResult.providers;
      providerMetadataDegraded = providerResult.degraded;
      discoveredModels = await fetchApertureModels(baseUrl, apiKey, logger, INTERACTIVE_FETCH_TIMEOUT_MS, discoveredProviders);
      warnProviderMetadataDegraded();
      return discoveredModels;
    }

    if (!refresh && modelLoadPromise) {
      return modelLoadPromise;
    }

    modelLoadPromise = waitForStableModels(baseUrl, apiKey, logger, {
      previousModels: discoveredModels,
      previousProviders: discoveredProviders,
    }).then((result) => {
      discoveredModels = result.models;
      discoveredProviders = result.providers;
      providerMetadataDegraded = result.providersDegraded;
      modelsLoaded = true;
      warnProviderMetadataDegraded();
      return discoveredModels;
    }).finally(() => {
      modelLoadPromise = undefined;
    });

    return modelLoadPromise;
  }

  function mutateConfig(config: Config): number {
    config.provider ??= {};

    if (discoveredModels.length === 0) {
      return 0;
    }

    const hadBaseProvider = Object.prototype.hasOwnProperty.call(config.provider, "aperture");
    const baseProvider = config.provider.aperture ?? {};
    const modelsByProvider = new Map<string, {
      group: ApertureProviderGroup;
      models: ApertureModel[];
    }>();

    for (const model of discoveredModels) {
      const group = getProviderGroup(model, discoveredProviders);
      const existingGroup = modelsByProvider.get(group.id);
      if (existingGroup) {
        existingGroup.models.push(model);
      } else {
        modelsByProvider.set(group.id, {
          group,
          models: [model],
        });
      }
    }

    for (const { group, models } of modelsByProvider.values()) {
      const existingProvider = config.provider[group.id] ?? {};
      const modelsObj: Record<string, ApertureModelConfig> = {
        ...(existingProvider.models as Record<string, ApertureModelConfig> ?? {}),
      };

      const npm = existingProvider.npm
        ?? (group.wireAPI === "openai" ? baseProvider.npm : undefined)
        ?? getProviderNpmPackage(group.wireAPI, group.compatibility);

      config.provider[group.id] = {
        ...baseProvider,
        ...existingProvider,
        npm,
        name: existingProvider.name ?? group.name,
        options: {
          ...baseProvider.options,
          ...existingProvider.options,
          baseURL: baseUrl,
          apiKey: existingProvider.options?.apiKey ?? baseProvider.options?.apiKey ?? apiKey,
        },
        models: modelsObj,
      };

      for (const model of models) {
        const existingModel = modelsObj[model.id] ?? {};
        const routeModelID = getApertureRouteModelID(model, discoveredProviders);
        const modelDefaults = getModelDefaults(model, modelsDevCatalog, discoveredProviders);
        if (!modelDefaults.matchedModelsDev && !warnedModelsDevFallbacks.has(routeModelID)) {
          warnedModelsDevFallbacks.add(routeModelID);
          logger.warn(`[TailscaleAperture] Model ${routeModelID} could not be matched to Models.dev specs; using conservative defaults`);
        }

        modelsObj[model.id] = {
          ...mergeModelConfig(modelDefaults.defaults, existingModel),
          id: existingModel.id ?? routeModelID,
          name: existingModel.name ?? model.id,
        };
      }
    }

    for (const providerID of Object.keys(config.provider)) {
      if (providerID.startsWith("aperture-") && !modelsByProvider.has(providerID)) {
        delete config.provider[providerID];
      }
    }

    const hasDefaultGroup = modelsByProvider.has("aperture");
    if (!hasDefaultGroup && !hadBaseProvider) {
      delete config.provider.aperture;
    }

    return modelsByProvider.size;
  }

  function countProviderGroups(models: ApertureModel[]): number {
    return new Set(models.map((model) => getProviderGroup(model, discoveredProviders).id)).size;
  }

  async function loadModelsOnStartup(): Promise<ApertureModel[]> {
    try {
      discoveredModels = await loadModels(false);
      if (discoveredModels.length === 0) {
        logger.warn("[TailscaleAperture] No models found");
        showMessage("success", `No Aperture models found at ${baseUrl}`);
        return discoveredModels;
      }

      logger.log(`[TailscaleAperture] Discovered ${discoveredModels.length} models from ${baseUrl}`);
      const providerGroupCount = countProviderGroups(discoveredModels);
      logger.log(`[TailscaleAperture] Registered ${providerGroupCount} Aperture provider groups for ${discoveredModels.length} discovered models`);
      showMessage("success", `Registered ${discoveredModels.length} Aperture models across ${providerGroupCount} provider groups`);
      return discoveredModels;
    } catch (error) {
      const errmsg = formatError(error);
      logger.error("[TailscaleAperture] Failed to register models:", error);
      showMessage("error", errmsg);
      await printErrorToChat(errmsg);
      throw error;
    }
  }

  const startupStartedAt = Date.now();
  let startupModelsDurationMs: number | undefined;
  let startupModelsDevDurationMs: number | undefined;

  const startupModels = (async () => {
    const startedAt = Date.now();
    try {
      return await loadModelsOnStartup();
    } finally {
      startupModelsDurationMs = Date.now() - startedAt;
      logger.info(`[TailscaleAperture] Startup step Aperture model discovery finished in ${formatDuration(startupModelsDurationMs)}`);
    }
  })();

  const startupModelsDevCatalog = (async () => {
    const startedAt = Date.now();
    try {
      const catalog = await loadModelsDevCatalog(modelsDevConfig, logger);
      modelsDevCatalog = catalog;
      if (catalog) {
        logger.log(`[TailscaleAperture] Loaded Models.dev catalog with ${Object.keys(catalog).length} providers`);
      }
      return catalog;
    } finally {
      startupModelsDevDurationMs = Date.now() - startedAt;
      logger.info(`[TailscaleAperture] Startup step Models.dev catalog load finished in ${formatDuration(startupModelsDevDurationMs)}`);
    }
  })();

  return {
    config: async (config: Config) => {
      const configWaitStartedAt = Date.now();
      try {
        await Promise.all([startupModels, startupModelsDevCatalog]);
        const configWaitDurationMs = Date.now() - configWaitStartedAt;
        const startupDurationMs = Date.now() - startupStartedAt;
        logger.info(`[TailscaleAperture] Startup finished in ${formatDuration(startupDurationMs)} (Aperture models: ${formatDuration(startupModelsDurationMs)}, Models.dev catalog: ${formatDuration(startupModelsDevDurationMs)}, config wait: ${formatDuration(configWaitDurationMs)})`);
        mutateConfig(config);
      } catch (error) {
        logger.error("[TailscaleAperture] Failed to register models:", error);
        showMessage("error", formatError(error));
      } finally {
        markTuiReady();
      }
    },

    event: async ({ event }) => {
      if (event.type === "server.connected") {
        markTuiReady();
      }
    },

    tool: {
      list_aperture_models: tool({
        description: "List available models from Tailscale Aperture",
        args: {
          refresh: tool.schema.boolean().optional().describe("Refresh the cached Aperture model list before returning it"),
        },
        async execute(args) {
          try {
            const models = await loadModels(args.refresh ?? false);
            return JSON.stringify({
              models,
              count: models.length,
            }, null, 2);
          } catch (error) {
            return JSON.stringify({ error: String(error) });
          }
        },
      }),

      get_aperture_model: tool({
        description: "Get details for a specific Aperture model",
        args: {
          modelId: tool.schema.string().describe("Model ID"),
          refresh: tool.schema.boolean().optional().describe("Refresh the cached Aperture model list before looking up the model"),
        },
        async execute(args) {
          try {
            const models = await loadModels(args.refresh ?? false);
            const model = models.find(m => m.id === args.modelId);
            if (!model) {
              return JSON.stringify({ error: `Model ${args.modelId} not found` });
            }
            return JSON.stringify({ model }, null, 2);
          } catch (error) {
            return JSON.stringify({ error: String(error) });
          }
        },
      }),
    },
  };
};

export default TailscaleAperturePlugin;
