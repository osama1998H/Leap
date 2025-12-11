# Web UI Implementation Guide

This document provides a step-by-step workflow for implementing features in the Leap Trading System Web UI. Follow this process to ensure complete, well-informed implementations.

---

## AI/Developer Implementation Prompt

Use the following prompt when starting work on any UI feature:

---

### Implementation Workflow Prompt

```
You are implementing features for the Leap Trading System Web UI. Follow this systematic workflow to gather all necessary information before writing code.

## Step 1: Check Implementation Status (ROADMAP.md)

First, read `docs/ui/ROADMAP.md` to understand:
1. Which phase we are currently in (Phase 1-4)
2. What tasks have been completed (marked with [x])
3. What tasks are pending (marked with [ ])
4. Dependencies between tasks
5. What features should be implemented next based on priority

Determine: "What feature am I implementing and what phase does it belong to?"

## Step 2: Understand Requirements (REQUIREMENTS.md)

Read `docs/ui/REQUIREMENTS.md` to find:
1. The relevant User Stories (US-R*, US-E*, US-T*) for your feature
2. The Functional Requirements (FR*) that describe what the feature must do
3. The Non-Functional Requirements (NFR*) for performance, accessibility, etc.
4. Any constraints or out-of-scope items to be aware of
5. Success metrics that define when the feature is complete

Extract: "What are the acceptance criteria for this feature?"

## Step 3: Map CLI to UI (CLI_MAPPING.md)

Read `docs/ui/CLI_MAPPING.md` to understand:
1. Which CLI command(s) this feature relates to
2. All CLI options/flags that need UI equivalents
3. Config-only options that should be in Advanced settings
4. Default values and validation rules for each field
5. Complex workflows that the UI should simplify

Document: "What inputs does the user need to provide and what are their constraints?"

## Step 4: Review Technical Architecture (ARCHITECTURE.md)

Read `docs/ui/ARCHITECTURE.md` to understand:
1. The overall system architecture and data flow
2. Which API endpoints you'll need to call
3. WebSocket events for real-time features
4. State management approach (Zustand stores, TanStack Query)
5. Error handling patterns
6. Security considerations

Plan: "How will data flow from UI to backend and back?"

## Step 5: Review Component Specifications (COMPONENTS.md)

Read `docs/ui/COMPONENTS.md` to find:
1. The page specification for your feature
2. Required form fields with types and validation
3. Data requirements (what API data is needed)
4. Shared components to reuse
5. Layout and section organization

Design: "What components do I need and how do they compose together?"

## Step 6: Check Wireframes (WIREFRAMES.md)

Read `docs/ui/WIREFRAMES.md` to see:
1. The visual layout for your feature's page(s)
2. Component placement and hierarchy
3. Information density and grouping
4. Mobile/responsive considerations

Visualize: "What should the user see and interact with?"

## Step 7: Identify UI Components (SHADCN_COMPONENTS.md)

Read `docs/ui/SHADCN_COMPONENTS.md` to determine:
1. Which shadcn/ui components to use
2. Installation commands needed
3. Custom component patterns to follow
4. Theming and styling guidelines
5. Code examples for similar components

Prepare: "What UI primitives and patterns should I use?"

## Step 8: Review API Contracts (API_SPECIFICATION.md)

Read `docs/ui/API_SPECIFICATION.md` to find:
1. Exact endpoint URLs, methods, and parameters
2. Request body schemas
3. Response schemas and status codes
4. Error response formats
5. WebSocket message formats (if real-time)
6. Pagination and filtering patterns

Integrate: "What are the exact API contracts I need to implement against?"

---

## Implementation Checklist

Before writing code, confirm you have answers to:

### Feature Definition
- [ ] What is the feature name and which phase is it from?
- [ ] What user stories does it fulfill?
- [ ] What are the functional requirements?

### User Interface
- [ ] What page(s) does this feature live on?
- [ ] What form fields/inputs are needed?
- [ ] What validation rules apply?
- [ ] What default values should be used?
- [ ] What loading/error states are needed?

### Data Flow
- [ ] What API endpoints does this feature call?
- [ ] What is the request/response format?
- [ ] Does this feature need real-time updates (WebSocket)?
- [ ] What state needs to be managed?

### Components
- [ ] What shadcn/ui components are needed?
- [ ] Are there custom components to build?
- [ ] What shared components can be reused?

### Testing
- [ ] What are the success metrics?
- [ ] What edge cases should be handled?
- [ ] What accessibility requirements apply?

---

## Example: Implementing "Training Configuration Page"

### Step 1: Check Roadmap
From ROADMAP.md → Phase 1 task F1.5: "Training config form" - Priority: Critical

### Step 2: Requirements
From REQUIREMENTS.md:
- US-R1: Configure training without memorizing CLI flags
- US-R4: Save and reuse configurations
- FR1.1-FR1.10: Training management requirements
- NFR3.4: Real-time form validation

### Step 3: CLI Mapping
From CLI_MAPPING.md:
- Maps to `train` command
- Fields: symbol(s), timeframe, bars, epochs, timesteps, model-dir
- Transformer config: d_model, n_heads, learning_rate, etc.
- PPO config: learning_rate, gamma, clip_epsilon, etc.
- Validation: epochs 1-1000, bars 1000-500000, etc.

### Step 4: Architecture
From ARCHITECTURE.md:
- Calls POST /api/v1/training/start
- Uses trainingStore (Zustand) for form state
- TanStack Query for config templates
- WebSocket subscription after job starts

### Step 5: Components
From COMPONENTS.md:
- Page: Training Configuration (/training)
- Sections: Data, Transformer, PPO, Advanced
- Components: TrainingConfigForm, SymbolSelector, TimeframeSelector
- Data: Config templates, validation schemas

### Step 6: Wireframes
From WIREFRAMES.md:
- Collapsible sections for each config category
- Summary panel showing key settings
- Preset selector in header
- Large "Start Training" CTA at bottom

### Step 7: UI Components
From SHADCN_COMPONENTS.md:
- Form, Input, Select, Switch, Slider, Tabs
- Card for sections
- Button for submit
- Toast for notifications

### Step 8: API
From API_SPECIFICATION.md:
- POST /training/start with full config body
- GET /config/templates for presets
- POST /config/validate for validation
- Response: { jobId, status, createdAt }

---

## Code Generation Guidelines

When implementing, follow these patterns:

### File Structure
```
src/
├── app/[feature]/page.tsx       # Page component
├── components/[feature]/        # Feature-specific components
├── hooks/use[Feature].ts        # Custom hooks
├── stores/[feature]Store.ts     # Zustand store
└── lib/schemas/[feature].ts     # Zod validation schemas
```

### Component Pattern
```tsx
// 1. Import shadcn components
// 2. Import custom hooks
// 3. Define props interface
// 4. Implement component with:
//    - Form validation (Zod + React Hook Form)
//    - Loading states (Skeleton)
//    - Error handling (Toast)
//    - Accessibility (ARIA labels)
```

### State Management Pattern
```tsx
// Zustand for UI state
const useFeatureStore = create<FeatureState>((set) => ({
  // State
  // Actions
}))

// TanStack Query for server state
const { data, isLoading } = useQuery({
  queryKey: ['feature', id],
  queryFn: () => api.getFeature(id)
})
```

### API Integration Pattern
```tsx
// Use the API client from lib/api.ts
// Handle errors with try/catch
// Show loading states
// Invalidate queries on mutations
```

---

## Quality Checklist

Before marking a feature complete:

- [ ] All form fields match CLI_MAPPING.md specifications
- [ ] Validation rules are implemented per requirements
- [ ] Loading and error states are handled
- [ ] Responsive design works on tablet
- [ ] Keyboard navigation works
- [ ] ARIA labels are present
- [ ] Dark mode works correctly
- [ ] API integration matches API_SPECIFICATION.md
- [ ] Success metrics from REQUIREMENTS.md are met
```

---

## Quick Reference: Document Reading Order

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE IMPLEMENTATION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │ ROADMAP.md   │ ──► "What should I build next?"                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ REQUIREMENTS.md  │ ──► "What are the acceptance criteria?"       │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ CLI_MAPPING.md   │ ──► "What inputs and validations are needed?" │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ ARCHITECTURE.md  │ ──► "How does data flow through the system?"  │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ COMPONENTS.md    │ ──► "What UI components do I need?"           │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ WIREFRAMES.md    │ ──► "What should the layout look like?"       │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────┐                                         │
│  │ SHADCN_COMPONENTS.md   │ ──► "What UI primitives should I use?"  │
│  └──────┬─────────────────┘                                         │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────┐                                         │
│  │ API_SPECIFICATION.md   │ ──► "What API contracts do I follow?"   │
│  └──────┬─────────────────┘                                         │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  IMPLEMENT CODE  │                                               │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Condensed AI Prompt (Copy-Paste Ready)

```
I need to implement a UI feature for the Leap Trading System. Follow this process:

1. Read docs/ui/ROADMAP.md - Find the feature in the task list, note its phase and priority
2. Read docs/ui/REQUIREMENTS.md - Extract user stories (US-*) and requirements (FR-*) for this feature
3. Read docs/ui/CLI_MAPPING.md - Get all form fields, defaults, and validation rules
4. Read docs/ui/ARCHITECTURE.md - Understand data flow, API endpoints, and state management
5. Read docs/ui/COMPONENTS.md - Get component specifications and page layout
6. Read docs/ui/WIREFRAMES.md - See the visual layout and structure
7. Read docs/ui/SHADCN_COMPONENTS.md - Identify shadcn/ui components and patterns to use
8. Read docs/ui/API_SPECIFICATION.md - Get exact API request/response schemas

After reading all documents, implement the feature with:
- Complete form validation matching CLI_MAPPING.md
- Proper loading/error states
- API integration per API_SPECIFICATION.md
- Components from SHADCN_COMPONENTS.md
- Layout matching WIREFRAMES.md

Mark the task as complete in ROADMAP.md when done.
```
