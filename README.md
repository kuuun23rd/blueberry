# AI Interview Simulator

An AI-powered mock interview app: candidates upload their CV, then talk face-to-face (two-way video) with a human-like AI interviewer avatar that asks CV-grounded questions and adapts its follow-ups based on what you actually say.

**Stack:** [LiveKit](https://livekit.io) (real-time video/audio room + Agents framework) + [Beyond Presence](https://www.beyondpresence.ai) (AI avatar, plugs into the LiveKit room)
**Go-live target:** 27.09.2026

---

## Repo structure

```
blueberry/
  backend/
    my-agent/        # Python — LiveKit Agents worker (STT → LLM → TTS pipeline, avatar wiring)
  frontend/           # Next.js/TypeScript — the web app candidates use to join a session
```

---

## Prerequisites (install these first)

| Tool | Used for | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python package/dependency manager for the backend | `brew install uv` |
| Python 3.10–3.14 | Backend runtime | usually handled by `uv` automatically |
| [pnpm](https://pnpm.io) | Package manager for the frontend | `brew install pnpm` |
| Node.js 24.x | Frontend runtime | `brew install node` (or use `nvm`) |
| [LiveKit CLI (`lk`)](https://docs.livekit.io/home/cli/cli-setup/) | Auth + pulling project env vars | `brew install livekit-cli` |

You'll also need to be added as a member of the team's **LiveKit Cloud project** and the **Beyond Presence** account — ask whoever set those up (see the team's PRD/implementation plan doc) to invite you, since the API keys below come from there.

---

## After cloning: backend setup

```bash
cd backend/my-agent
cp .env.example .env.local
```

Fill in `.env.local` with the project's real values (don't commit this file — it's already gitignored):
```
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```
If you're authenticated via `lk cloud auth` and linked to the team's project, you can generate this automatically instead of typing it by hand:
```bash
lk app env --write --destination .env.local
```

Install dependencies and test it runs:
```bash
uv sync
uv run python src/agent.py console
```
`console` mode lets you talk to the agent right in your terminal (no frontend needed) — good for a first smoke test.

---

## After cloning: frontend setup

```bash
cd frontend
cp .env.example .env.local
```

Fill in the same LiveKit values in `frontend/.env.local`:
```
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
AGENT_NAME=
```
Leave `AGENT_NAME`, `NEXT_PUBLIC_APP_CONFIG_ENDPOINT`, and `SANDBOX_ID` blank — the last two are for LiveKit's internal tooling, not needed here.

Install dependencies:
```bash
pnpm install
```

---

## Running the full app locally

You need **both** the backend agent and the frontend running at the same time, in two separate terminal tabs:

```bash
# terminal 1
cd backend/my-agent
uv run python src/agent.py dev

# terminal 2
cd frontend
pnpm dev
```

Then open **http://localhost:3000** — you should be able to join a session and talk to the agent live.

---

## Where things stand / what's not built yet

- The backend agent is currently the generic LiveKit starter — no CV upload, no CV-grounded questions, no Beyond Presence avatar wired in yet. Those are upcoming milestones (see the implementation plan doc).
- Don't commit `.env.local` in either folder — it's already in `.gitignore`. Never share API keys in chat/commits; use whatever the team has agreed on for sharing secrets (e.g. a password manager).
