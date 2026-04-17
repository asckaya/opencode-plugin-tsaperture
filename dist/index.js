import { tool } from "@opencode-ai/plugin";
import { homedir } from "os";
import { join } from "path";
import { readFile } from "fs/promises";
import { platform } from "process";
function normalizeBaseUrl(baseUrl) {
    return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}
function slugifyProviderSegment(value) {
    const normalized = value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
    return normalized || "default";
}
function getProviderGroup(model) {
    const providerID = model.metadata?.provider?.id?.trim();
    const providerName = model.metadata?.provider?.name?.trim();
    if (!providerName) {
        return {
            id: "aperture",
            name: "Aperture",
        };
    }
    return {
        id: `aperture-${slugifyProviderSegment(providerID || providerName)}`,
        name: `Aperture/${providerName}`,
    };
}
function getModelDefaults(model) {
    const id = model.id.toLowerCase();
    const providerID = model.metadata?.provider?.id?.toLowerCase();
    const providerName = model.metadata?.provider?.name?.toLowerCase();
    const isZai = id.includes("glm")
        || providerID === "zai"
        || providerID === "z.ai"
        || providerID === "zai-coding-plan"
        || providerName === "z.ai"
        || providerName === "zai-coding-plan";
    const isKimi = id.includes("kimi")
        || providerID === "kimi"
        || providerID === "kimi-for-coding"
        || providerName === "kimi"
        || providerName === "kimi-for-coding";
    if (isZai) {
        return {
            limit: {
                context: 200_000,
                output: 8_192,
            },
            reasoning: true,
            temperature: true,
            tool_call: true,
            modalities: {
                input: ["text"],
                output: ["text"],
            },
            interleaved: {
                field: "reasoning_content",
            },
            options: {
                thinking: {
                    type: "enabled",
                    clear_thinking: false,
                },
            },
        };
    }
    if (isKimi) {
        return {
            limit: {
                context: 200_000,
                output: 128_000,
            },
            reasoning: true,
            temperature: true,
            tool_call: true,
            modalities: {
                input: ["text"],
                output: ["text"],
            },
            interleaved: {
                field: "reasoning_content",
            },
            options: {
                thinking: {
                    type: "enabled",
                    clear_thinking: false,
                },
            },
            headers: {
                "User-Agent": "KimiCLI/1.3",
            },
        };
    }
    return {
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
        options: {
            thinking: {
                type: "enabled",
                clear_thinking: false,
            },
        },
    };
}
function getDefaultReleaseDate(created) {
    if (!created || created <= 0) {
        return "";
    }
    return new Date(created * 1000).toISOString().slice(0, 10);
}
function mergeThinkingConfig(defaults, existing) {
    if (!defaults && !existing) {
        return undefined;
    }
    return {
        ...defaults,
        ...existing,
    };
}
function mergeModelConfig(defaults, existing = {}) {
    const thinking = mergeThinkingConfig(defaults.options?.thinking, existing.options?.thinking);
    const limit = defaults.limit || existing.limit ? {
        context: existing.limit?.context ?? defaults.limit?.context ?? 0,
        input: existing.limit?.input ?? defaults.limit?.input,
        output: existing.limit?.output ?? defaults.limit?.output ?? 0,
    } : undefined;
    const modalities = defaults.modalities || existing.modalities ? {
        input: existing.modalities?.input ?? defaults.modalities?.input ?? ["text"],
        output: existing.modalities?.output ?? defaults.modalities?.output ?? ["text"],
    } : undefined;
    return {
        ...defaults,
        ...existing,
        ...(limit ? { limit } : {}),
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
    };
}
/**
 * Poll the models endpoint until the set of model IDs stabilizes (two
 * consecutive fetches return the same IDs) or the deadline is exceeded.
 * Transient fetch errors are retried within the deadline.
 */
async function waitForStableModels(baseUrl, apiKey, { pollIntervalMs = 500, deadlineMs = 10_000, minFetchTimeoutMs = 2_000, previousModels = [] } = {}) {
    const deadline = Date.now() + deadlineMs;
    let previousIds = previousModels.length > 0
        ? previousModels.map((m) => m.id).sort().join("\n")
        : undefined;
    let lastGoodResult = previousModels;
    while (Date.now() < deadline) {
        const remaining = deadline - Date.now();
        if (remaining < minFetchTimeoutMs && lastGoodResult.length > 0) {
            return lastGoodResult;
        }
        try {
            const models = await fetchApertureModels(baseUrl, apiKey, Math.max(remaining, minFetchTimeoutMs));
            const ids = models.map((m) => m.id).sort().join("\n");
            lastGoodResult = models;
            if (ids === previousIds) {
                return models;
            }
            previousIds = ids;
        }
        catch {
            // Transient error — retry until deadline.
        }
        if (Date.now() + pollIntervalMs >= deadline) {
            return lastGoodResult;
        }
        await new Promise((r) => setTimeout(r, pollIntervalMs));
    }
    return lastGoodResult;
}
async function fetchApertureModels(baseUrl, apiKey, timeoutMs = 15_000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(`${baseUrl}/v1/models`, {
            signal: controller.signal,
            headers: apiKey ? {
                Authorization: `Bearer ${apiKey}`,
            } : undefined,
        });
        if (!response.ok) {
            throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
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
        return Array.from(new Map(mergedModels.map((model) => [model.id, model])).values());
    }
    finally {
        clearTimeout(timer);
    }
}
function getOpenCodeConfigDirs() {
    const home = homedir();
    const dirs = [];
    if (platform === "win32") {
        dirs.push(join(process.env.APPDATA || process.env.LOCALAPPDATA || home, "opencode"));
    }
    else if (platform === "darwin") {
        const xdgConfig = process.env.XDG_CONFIG_HOME;
        if (xdgConfig) {
            dirs.push(join(xdgConfig, "opencode"));
        }
        dirs.push(join(home, ".config", "opencode"));
        dirs.push(join(home, "Library", "Application Support", "opencode"));
    }
    else {
        const xdgConfig = process.env.XDG_CONFIG_HOME;
        if (xdgConfig) {
            dirs.push(join(xdgConfig, "opencode"));
        }
        dirs.push(join(home, ".config", "opencode"));
    }
    return dirs;
}
const openCodeConfigDirs = getOpenCodeConfigDirs();
async function loadApertureConfig() {
    for (const configDir of openCodeConfigDirs) {
        const configPath = join(configDir, "aperture.json");
        try {
            const content = await readFile(configPath, "utf-8");
            console.log(`[TailscaleAperture] Loaded config from ${configPath}`);
            return JSON.parse(content);
        }
        catch (error) {
            if (error.code !== "ENOENT") {
                console.warn(`[TailscaleAperture] Failed to read ${configPath}:`, error);
            }
        }
    }
    return {};
}
export const TailscaleAperturePlugin = async (_ctx, options) => {
    const fileConfig = await loadApertureConfig();
    const rawBaseUrl = options?.baseUrl || process.env.APERTURE_BASE_URL || fileConfig.baseUrl;
    const apiKey = options?.apiKey || process.env.APERTURE_API_KEY || fileConfig.apiKey || "";
    if (!rawBaseUrl) {
        console.warn("[TailscaleAperture] No baseUrl configured. Set APERTURE_BASE_URL, add baseUrl to plugin options, or create aperture.json in opencode config directory.");
        return {};
    }
    if (!apiKey) {
        console.info("[TailscaleAperture] No API key configured. This may be okay if you don't use authorization.");
    }
    const baseUrl = normalizeBaseUrl(rawBaseUrl);
    let discoveredModels = [];
    let modelsLoaded = false;
    async function loadModels(refresh = false) {
        if (!refresh && modelsLoaded) {
            return discoveredModels;
        }
        if (refresh && modelsLoaded) {
            // Interactive refresh: single fetch, no stabilization wait.
            discoveredModels = await fetchApertureModels(baseUrl, apiKey);
            return discoveredModels;
        }
        discoveredModels = await waitForStableModels(baseUrl, apiKey, {
            previousModels: discoveredModels,
        });
        modelsLoaded = true;
        return discoveredModels;
    }
    try {
        discoveredModels = await loadModels(true);
        if (discoveredModels.length === 0) {
            console.warn("[TailscaleAperture] No models found");
        }
        else {
            console.log(`[TailscaleAperture] Discovered ${discoveredModels.length} models from ${baseUrl}`);
        }
    }
    catch (error) {
        console.warn("[TailscaleAperture] Failed to preload models:", error);
    }
    return {
        config: async (config) => {
            try {
                config.provider ??= {};
                if (discoveredModels.length === 0) {
                    return;
                }
                const baseProvider = config.provider.aperture ?? {};
                const modelsByProvider = new Map();
                for (const model of discoveredModels) {
                    const group = getProviderGroup(model);
                    const existingGroup = modelsByProvider.get(group.id);
                    if (existingGroup) {
                        existingGroup.models.push(model);
                    }
                    else {
                        modelsByProvider.set(group.id, {
                            group,
                            models: [model],
                        });
                    }
                }
                for (const { group, models } of modelsByProvider.values()) {
                    const existingProvider = config.provider[group.id] ?? {};
                    const modelsObj = {
                        ...(existingProvider.models ?? {}),
                    };
                    config.provider[group.id] = {
                        ...baseProvider,
                        ...existingProvider,
                        npm: existingProvider.npm ?? baseProvider.npm ?? "@ai-sdk/openai-compatible",
                        name: existingProvider.name ?? group.name,
                        options: {
                            ...baseProvider.options,
                            ...existingProvider.options,
                            baseURL: `${baseUrl}/v1`,
                            apiKey: existingProvider.options?.apiKey ?? baseProvider.options?.apiKey ?? apiKey,
                        },
                        models: modelsObj,
                    };
                    for (const model of models) {
                        const existingModel = modelsObj[model.id] ?? {};
                        modelsObj[model.id] = {
                            ...mergeModelConfig(getModelDefaults(model), existingModel),
                            id: model.id,
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
                if (!hasDefaultGroup) {
                    delete config.provider.aperture;
                }
                console.log(`[TailscaleAperture] Registered ${modelsByProvider.size} Aperture provider groups for ${discoveredModels.length} discovered models`);
            }
            catch (error) {
                console.error("[TailscaleAperture] Failed to register models:", error);
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
                    }
                    catch (error) {
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
                    }
                    catch (error) {
                        return JSON.stringify({ error: String(error) });
                    }
                },
            }),
        },
    };
};
export default TailscaleAperturePlugin;
//# sourceMappingURL=index.js.map