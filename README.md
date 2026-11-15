# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/madara200599/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/madara200599/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One zero-dependency Python server that speaks both the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in streaming, and routes every request across ten free-tier providers (OpenAI-, Anthropic- and Gemini-wire) to the highest-rated free model that meets the request's requirements — with quality-ordered failover when a provider refuses, per-provider failure memory, and a catalog probed live from the providers themselves. Point any OpenAI or Anthropic client at it and it keeps answering while each provider's free tier lasts.

## Run it

```sh
python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tvfbjxfd&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
