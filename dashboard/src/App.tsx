import {
  AccountTree,
  BugReport,
  ChatBubbleOutline,
  DataObject,
  FileUpload,
  InfoOutlined,
  Insights,
  Memory,
  Search,
  Timeline,
} from '@mui/icons-material';
import {
  Alert,
  AppBar,
  Autocomplete,
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  List,
  ListItemButton,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material';
import { useEffect, useMemo, useState } from 'react';

type Role = 'user' | 'assistant' | 'tool';

type ToolCall = {
  endpoint: string;
  arguments: Record<string, unknown>;
};

type ArgumentSource = {
  source?: string;
  value?: unknown;
  evidence?: string;
  source_endpoint?: string | null;
};

type ToolStepTrace = {
  step: number;
  endpoint: string;
  goal?: string;
  depends_on?: string[];
  argument_sources?: Record<string, ArgumentSource>;
  output_refs?: Record<string, unknown>;
  status?: string;
};

type Message = {
  role: Role;
  content: string | Record<string, unknown> | null;
  tool_calls?: ToolCall[] | null;
};

type ScoreDimension = {
  score: number;
  rationale?: string;
};

type JudgeScores = {
  tool_correctness?: ScoreDimension;
  naturalness?: ScoreDimension;
  task_completion?: ScoreDimension;
  overall?: number | ScoreDimension;
};

type ScenarioPlan = {
  scenario?: string;
  user_persona?: string;
  expected_tool_sequence?: string[];
  disambiguation_points?: string[];
  complexity?: string;
};

type Metadata = {
  seed?: number;
  conversation_index?: number;
  tools_used?: string[];
  num_turns?: number;
  num_tool_calls?: number;
  num_distinct_tools?: number;
  pattern?: string;
  category_domains?: string[];
  chain_source?: string;
  repair_attempts?: number;
  steering_enabled?: boolean;
  generation_timestamp?: string;
  model?: string;
  generation_profile?: string;
  planner_scenario?: ScenarioPlan;
};

type Conversation = {
  conversation_id: string;
  messages: Message[];
  step_trace?: ToolStepTrace[];
  judge_scores?: JudgeScores;
  metadata?: Metadata;
};

type Endpoint = {
  tool_name: string;
  endpoint_name: string;
  endpoint_id: string;
  description?: string;
  method?: string;
  category?: string;
  parameters?: Array<{
    name: string;
    type?: string;
    description?: string;
    required?: boolean;
    default?: unknown;
  }>;
  response_schema?: unknown;
};

type Edge = {
  source: string;
  target: string;
  edge_type: string;
  edge_types?: string[];
  weight?: number;
};

type DiversityRun = {
  tool_combination_entropy?: number;
  domain_coverage_cv?: number;
  unique_tool_pairs?: number;
  unique_domains_used?: number;
  total_domains_available?: number;
  pattern_distribution?: Record<string, number>;
  mean_quality_score?: number;
};

type DashboardBundle = {
  source?: { dataset?: string; artifacts_dir?: string };
  conversations: Conversation[];
  liveSamples?: Conversation[];
  runA?: Conversation[];
  runB?: Conversation[];
  diversity?: {
    run_a?: DiversityRun;
    run_b?: DiversityRun;
    seed?: number;
    num_conversations?: number;
  };
  artifacts?: {
    registryStats?: Record<string, unknown>;
    graphStats?: Record<string, unknown>;
    endpoints?: Endpoint[];
    edges?: Edge[];
  };
};

type TraceKind = 'user' | 'assistant' | 'tool_call' | 'tool_response' | 'judge' | 'repair';
type TraceStatus = 'ok' | 'warn' | 'error' | 'info';

type TraceEvent = {
  id: string;
  turnIndex: number;
  order: number;
  kind: TraceKind;
  status: TraceStatus;
  label: string;
  subtitle: string;
  message?: Message;
  endpoint?: string;
  args?: Record<string, unknown>;
  response?: unknown;
  previousEndpoint?: string;
  relation?: Edge;
  groundedArgs?: string[];
  stepTrace?: ToolStepTrace;
  score?: number;
  payload: Record<string, unknown>;
};

const emptyBundle: DashboardBundle = {
  conversations: [],
  liveSamples: [],
  runA: [],
  runB: [],
  diversity: {},
  artifacts: {
    endpoints: [],
    edges: [],
  },
};

const eventColors: Record<TraceStatus, string> = {
  ok: '#2e7d32',
  warn: '#ed6c02',
  error: '#d32f2f',
  info: '#1976d2',
};

const edgeColors: Record<string, string> = {
  io_chain: '#1976d2',
  complementary: '#2e7d32',
  same_tool: '#6f42c1',
  same_category: '#6b7280',
};

function App() {
  const [bundle, setBundle] = useState<DashboardBundle>(emptyBundle);
  const [loadError, setLoadError] = useState('');
  const [query, setQuery] = useState('');
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [toolMatchMode, setToolMatchMode] = useState<'any' | 'all'>('any');
  const [selectedId, setSelectedId] = useState('');
  const [selectedEventId, setSelectedEventId] = useState('');
  const [hoveredTurn, setHoveredTurn] = useState<number | null>(null);
  const [detailTab, setDetailTab] = useState(0);

  useEffect(() => {
    fetch('/toolgen-data/bundle.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data: DashboardBundle) => {
        const normalized = normalizeBundle(data);
        setBundle(normalized);
        setSelectedId(normalized.conversations[0]?.conversation_id ?? '');
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  const conversations = bundle.conversations ?? [];
  const endpoints = bundle.artifacts?.endpoints ?? [];
  const edges = bundle.artifacts?.edges ?? [];
  const endpointMap = useMemo(() => new Map(endpoints.map((ep) => [ep.endpoint_id, ep])), [endpoints]);
  const edgeMap = useMemo(() => {
    const map = new Map<string, Edge>();
    edges.forEach((edge) => map.set(edgeKey(edge.source, edge.target), edge));
    return map;
  }, [edges]);

  const toolOptions = useMemo(() => collectToolOptions(conversations), [conversations]);
  const filteredConversations = useMemo(
    () => filterConversations(conversations, query, selectedTools, toolMatchMode),
    [conversations, query, selectedTools, toolMatchMode],
  );
  const metrics = useMemo(() => computeMetrics(filteredConversations), [filteredConversations]);

  useEffect(() => {
    if (!filteredConversations.length) {
      setSelectedId('');
      return;
    }
    if (!filteredConversations.some((conversation) => conversation.conversation_id === selectedId)) {
      setSelectedId(filteredConversations[0].conversation_id);
    }
  }, [filteredConversations, selectedId]);

  const selectedConversation =
    filteredConversations.find((conversation) => conversation.conversation_id === selectedId) ??
    filteredConversations[0] ??
    null;

  const traceEvents = useMemo(
    () => (selectedConversation ? buildTraceEvents(selectedConversation, edgeMap) : []),
    [selectedConversation, edgeMap],
  );

  useEffect(() => {
    if (!traceEvents.length) {
      setSelectedEventId('');
      return;
    }
    if (!traceEvents.some((event) => event.id === selectedEventId)) {
      setSelectedEventId(traceEvents.find((event) => event.kind === 'tool_call')?.id ?? traceEvents[0].id);
    }
  }, [traceEvents, selectedEventId]);

  const selectedEvent =
    traceEvents.find((event) => event.id === selectedEventId) ??
    traceEvents[0] ??
    null;

  const handleConversationUpload = async (file: File | null) => {
    if (!file) {
      return;
    }
    const parsed = parseJsonl(await file.text());
    setBundle((current) => normalizeBundle({ ...current, conversations: parsed }));
    setSelectedId(parsed[0]?.conversation_id ?? '');
  };

  const handleBundleUpload = async (file: File | null) => {
    if (!file) {
      return;
    }
    const parsed = JSON.parse(await file.text()) as DashboardBundle;
    const normalized = normalizeBundle(parsed);
    setBundle(normalized);
    setSelectedId(normalized.conversations[0]?.conversation_id ?? '');
  };

  return (
    <Box className="appShell">
      <AppBar position="static" color="inherit" elevation={0} className="topBar">
        <Toolbar className="toolbar">
          <Stack direction="row" alignItems="center" spacing={1.25} className="titleBlock">
            <DataObject className="brandIcon" />
            <Box className="titleCopy">
              <Typography variant="h1">ToolGen Trace Workbench</Typography>
              <Typography variant="body2" color="text.secondary">
                {bundle.source?.dataset ?? 'Load a ToolGen JSONL or dashboard bundle'}
              </Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={0.75} className="metricRail">
            <TopMetric label="Chats" value={metrics.totalConversations} tone="cyan" />
            <TopMetric label="Calls" value={metrics.totalToolCalls} tone="green" />
            <TopMetric label="Mean" value={formatNumber(metrics.meanScore, 2)} tone="pink" />
            <TopMetric label="Review" value={metrics.reviewCount} tone="yellow" />
          </Stack>

          <Stack direction="row" spacing={1} className="toolbarActions">
            <Button component="label" variant="outlined" size="small" startIcon={<FileUpload />}>
              JSONL
              <input
                hidden
                type="file"
                accept=".jsonl,.txt"
                onChange={(event) => void handleConversationUpload(event.target.files?.[0] ?? null)}
              />
            </Button>
            <Button component="label" variant="contained" size="small" startIcon={<FileUpload />}>
              Bundle
              <input
                hidden
                type="file"
                accept=".json"
                onChange={(event) => void handleBundleUpload(event.target.files?.[0] ?? null)}
              />
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>

      {loadError && (
        <Alert severity="warning" className="loadWarning">
          Default dashboard bundle was not loaded: {loadError}
        </Alert>
      )}

      <Box component="main" className="workbench">
        <ConversationListPane
          conversations={filteredConversations}
          allConversations={conversations}
          selectedId={selectedConversation?.conversation_id ?? ''}
          query={query}
          selectedTools={selectedTools}
          toolOptions={toolOptions}
          toolMatchMode={toolMatchMode}
          onQueryChange={setQuery}
          onSelectedToolsChange={setSelectedTools}
          onToolMatchModeChange={setToolMatchMode}
          onSelectConversation={(id) => {
            setSelectedId(id);
            setSelectedEventId('');
            setDetailTab(0);
          }}
        />

        <ChatPane
          conversation={selectedConversation}
          traceEvents={traceEvents}
          selectedEventId={selectedEventId}
          hoveredTurn={hoveredTurn}
          onSelectEvent={(event) => {
            setSelectedEventId(event.id);
            setDetailTab(event.kind === 'tool_call' ? 1 : 0);
          }}
        />

        <TracePane
          conversation={selectedConversation}
          traceEvents={traceEvents}
          selectedEvent={selectedEvent}
          endpointMap={endpointMap}
          detailTab={detailTab}
          hoveredTurn={hoveredTurn}
          onDetailTabChange={setDetailTab}
          onSelectEvent={(event) => {
            setSelectedEventId(event.id);
            setDetailTab(event.kind === 'tool_call' ? 1 : 0);
          }}
          onHoverTurn={setHoveredTurn}
        />
      </Box>
    </Box>
  );
}

function TopMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | number;
  tone: 'cyan' | 'green' | 'pink' | 'yellow';
}) {
  return (
    <Box className={`topMetric topMetric-${tone}`}>
      <Typography className="topMetricValue">{value}</Typography>
      <Typography className="topMetricLabel">{label}</Typography>
    </Box>
  );
}

function ConversationListPane({
  conversations,
  allConversations,
  selectedId,
  query,
  selectedTools,
  toolOptions,
  toolMatchMode,
  onQueryChange,
  onSelectedToolsChange,
  onToolMatchModeChange,
  onSelectConversation,
}: {
  conversations: Conversation[];
  allConversations: Conversation[];
  selectedId: string;
  query: string;
  selectedTools: string[];
  toolOptions: string[];
  toolMatchMode: 'any' | 'all';
  onQueryChange: (query: string) => void;
  onSelectedToolsChange: (tools: string[]) => void;
  onToolMatchModeChange: (mode: 'any' | 'all') => void;
  onSelectConversation: (conversationId: string) => void;
}) {
  return (
    <Paper className="pane sessionPane">
      <Box className="paneHeader">
        <Stack direction="row" alignItems="center" spacing={1}>
          <ChatBubbleOutline fontSize="small" color="primary" />
          <Box>
            <Typography variant="h2">Chats</Typography>
            <Typography variant="caption" color="text.secondary">
              {conversations.length} / {allConversations.length} visible
            </Typography>
          </Box>
        </Stack>
      </Box>

      <Stack spacing={1} className="sessionFilters">
        <TextField
          size="small"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search chats, tools, domains"
          InputProps={{ startAdornment: <Search fontSize="small" className="searchIcon" /> }}
        />
        <Autocomplete
          multiple
          size="small"
          options={toolOptions}
          value={selectedTools}
          onChange={(_, value) => onSelectedToolsChange(value)}
          limitTags={1}
          renderInput={(params) => <TextField {...params} placeholder="Tool filter" />}
        />
        <FormControl size="small">
          <InputLabel>Tool match</InputLabel>
          <Select
            label="Tool match"
            value={toolMatchMode}
            onChange={(event) => onToolMatchModeChange(event.target.value as 'any' | 'all')}
          >
            <MenuItem value="any">Any selected tool</MenuItem>
            <MenuItem value="all">All selected tools</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      <List disablePadding className="sessionList">
        {conversations.map((conversation) => (
          <ListItemButton
            key={conversation.conversation_id}
            selected={conversation.conversation_id === selectedId}
            onClick={() => onSelectConversation(conversation.conversation_id)}
            className="sessionItem"
          >
            <ListItemText
              primary={
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                  <Typography className="sessionId">{conversation.conversation_id}</Typography>
                  <ScorePill score={overallScore(conversation)} />
                </Stack>
              }
              secondary={
                <Box className="sessionMeta">
                  <Typography component="span">
                    {conversation.metadata?.pattern ?? 'unknown'}
                  </Typography>
                  <Typography component="span">
                    {conversation.metadata?.num_tool_calls ?? countToolCalls(conversation)} calls
                  </Typography>
                  <Typography component="span">
                    {(conversation.metadata?.category_domains ?? ['unknown']).join(', ')}
                  </Typography>
                </Box>
              }
            />
          </ListItemButton>
        ))}
      </List>
    </Paper>
  );
}

function ChatPane({
  conversation,
  traceEvents,
  selectedEventId,
  hoveredTurn,
  onSelectEvent,
}: {
  conversation: Conversation | null;
  traceEvents: TraceEvent[];
  selectedEventId: string;
  hoveredTurn: number | null;
  onSelectEvent: (event: TraceEvent) => void;
}) {
  if (!conversation) {
    return (
      <Paper className="pane chatPane">
        <EmptyState title="No chat selected" body="Load a dataset or adjust filters." />
      </Paper>
    );
  }

  const scenario = conversation.metadata?.planner_scenario?.scenario;
  const toolPath = conversation.metadata?.planner_scenario?.expected_tool_sequence ??
    conversation.metadata?.tools_used ??
    [];

  return (
    <Paper className="pane chatPane">
      <Box className="chatHeader">
        <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={2}>
          <Box className="chatTitle">
            <Typography variant="h2">{conversation.conversation_id}</Typography>
            <Typography color="text.secondary" className="scenarioText">
              {scenario ?? 'No planner scenario was stored for this conversation.'}
            </Typography>
          </Box>
          <Stack direction="row" spacing={0.75} className="chatHeaderChips">
            <ScorePill score={overallScore(conversation)} />
            <Chip size="small" label={conversation.metadata?.generation_profile ?? 'profile n/a'} />
            <Chip size="small" label={conversation.metadata?.model ?? 'unknown model'} variant="outlined" />
          </Stack>
        </Stack>
        <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap className="pathStrip">
          {toolPath.length ? (
            toolPath.map((endpoint, index) => (
              <Chip
                key={`${endpoint}-${index}`}
                size="small"
                label={`${index + 1}. ${shortEndpoint(endpoint)}`}
                color="primary"
                variant="outlined"
              />
            ))
          ) : (
            <Chip size="small" label="No sampled path" variant="outlined" />
          )}
        </Stack>
      </Box>

      <Box className="chatScroll">
        {conversation.messages.map((message, turnIndex) => {
          if (message.role === 'tool') {
            return null;
          }
          const messageEvents = traceEvents.filter((event) => event.turnIndex === turnIndex);
          return (
            <ChatTurn
              key={`${conversation.conversation_id}-${turnIndex}`}
              message={message}
              turnIndex={turnIndex}
              events={messageEvents}
              selectedEventId={selectedEventId}
              isHighlighted={hoveredTurn === turnIndex || messageEvents.some((event) => event.id === selectedEventId)}
              onSelectEvent={onSelectEvent}
            />
          );
        })}
      </Box>
    </Paper>
  );
}

function ChatTurn({
  message,
  turnIndex,
  events,
  selectedEventId,
  isHighlighted,
  onSelectEvent,
}: {
  message: Message;
  turnIndex: number;
  events: TraceEvent[];
  selectedEventId: string;
  isHighlighted: boolean;
  onSelectEvent: (event: TraceEvent) => void;
}) {
  const text = formatContent(message.content);
  const toolEvents = events.filter((event) => event.kind === 'tool_call');
  const roleName = message.role === 'user' ? 'User' : 'Assistant';

  return (
    <Box className={`chatTurn ${message.role} ${isHighlighted ? 'chatTurn-active' : ''}`}>
      {text && (
        <Box className="chatBubble">
          <Stack direction="row" alignItems="center" justifyContent="space-between" className="bubbleTop">
            <Typography className="speaker">{roleName}</Typography>
            <Typography className="turnNumber">turn {turnIndex + 1}</Typography>
          </Stack>
          <Typography className="messageText">{text}</Typography>
        </Box>
      )}

      {toolEvents.length > 0 && (
        <Stack spacing={0.75} className={`toolCallRail ${text ? 'toolCallRail-under' : ''}`}>
          {toolEvents.map((event) => (
            <Button
              key={event.id}
              variant={event.id === selectedEventId ? 'contained' : 'outlined'}
              color={event.groundedArgs?.length ? 'secondary' : 'primary'}
              size="small"
              startIcon={<DataObject />}
              onClick={() => onSelectEvent(event)}
              className="toolCallButton"
            >
              {shortEndpoint(event.endpoint ?? 'tool_call')}
            </Button>
          ))}
        </Stack>
      )}
    </Box>
  );
}

function TracePane({
  conversation,
  traceEvents,
  selectedEvent,
  endpointMap,
  detailTab,
  hoveredTurn,
  onDetailTabChange,
  onSelectEvent,
  onHoverTurn,
}: {
  conversation: Conversation | null;
  traceEvents: TraceEvent[];
  selectedEvent: TraceEvent | null;
  endpointMap: Map<string, Endpoint>;
  detailTab: number;
  hoveredTurn: number | null;
  onDetailTabChange: (tab: number) => void;
  onSelectEvent: (event: TraceEvent) => void;
  onHoverTurn: (turn: number | null) => void;
}) {
  const endpoint = selectedEvent?.endpoint ? endpointMap.get(selectedEvent.endpoint) : undefined;
  const toolCallCount = traceEvents.filter((event) => event.kind === 'tool_call').length;
  const warningCount = traceEvents.filter((event) => event.status !== 'ok' && event.status !== 'info').length;

  return (
    <Paper className="pane tracePane">
      <Box className="paneHeader traceHeader">
        <Stack direction="row" alignItems="center" spacing={1}>
          <Timeline color="primary" />
          <Box>
            <Typography variant="h2">Events</Typography>
            <Typography variant="caption" color="text.secondary">
              ADK-style trace for selected chat
            </Typography>
          </Box>
        </Stack>
        <Stack direction="row" spacing={0.75}>
          <Chip size="small" label={`${traceEvents.length} events`} />
          <Chip size="small" color="primary" label={`${toolCallCount} calls`} />
          <Chip size="small" color={warningCount ? 'warning' : 'success'} label={`${warningCount} flags`} />
        </Stack>
      </Box>

      <Box className="traceList">
        {traceEvents.map((event) => (
          <TraceRow
            key={event.id}
            event={event}
            selected={selectedEvent?.id === event.id}
            hovered={hoveredTurn === event.turnIndex}
            onClick={() => onSelectEvent(event)}
            onMouseEnter={() => onHoverTurn(event.turnIndex)}
            onMouseLeave={() => onHoverTurn(null)}
          />
        ))}
      </Box>

      <Box className="eventDetails">
        {conversation && selectedEvent ? (
          <>
            <Box className="detailsHeader">
              <Stack direction="row" alignItems="center" spacing={1}>
                <EventGlyph kind={selectedEvent.kind} />
                <Box className="detailsTitle">
                  <Typography variant="h3">{selectedEvent.label}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {selectedEvent.subtitle}
                  </Typography>
                </Box>
              </Stack>
              <StatusDot status={selectedEvent.status} />
            </Box>

            <Tabs value={detailTab} onChange={(_, value) => onDetailTabChange(value)} className="detailTabs">
              <Tab label="Event" />
              <Tab label="Request" />
              <Tab label="Response" />
              <Tab label="Graph" />
            </Tabs>

            {detailTab === 0 && (
              <Stack spacing={1.25} className="detailBody">
                <KeyValue label="Turn" value={String(selectedEvent.turnIndex + 1)} />
                <KeyValue label="Kind" value={selectedEvent.kind} />
                <KeyValue label="Status" value={selectedEvent.status} />
                {selectedEvent.endpoint && <KeyValue label="Endpoint" value={selectedEvent.endpoint} />}
                {selectedEvent.relation && (
                  <KeyValue
                    label="Graph relation"
                    value={`${selectedEvent.relation.edge_type} (${formatNumber(selectedEvent.relation.weight ?? 0, 2)})`}
                  />
                )}
                <CodeBlock title="Raw Event" value={selectedEvent.payload} />
              </Stack>
            )}

            {detailTab === 1 && (
              <Stack spacing={1.25} className="detailBody">
                <KeyValue label="Previous endpoint" value={selectedEvent.previousEndpoint ?? 'none'} />
                <KeyValue
                  label="Grounded args"
                  value={selectedEvent.groundedArgs?.join(', ') || 'none detected'}
                />
                <CodeBlock title="Tool Arguments / Request" value={selectedEvent.args ?? selectedEvent.payload} />
                <CodeBlock title="Trace-First Reasoning" value={selectedEvent.stepTrace ?? {}} />
                <CodeBlock title="Endpoint Parameters" value={endpoint?.parameters ?? []} />
              </Stack>
            )}

            {detailTab === 2 && (
              <Stack spacing={1.25} className="detailBody">
                <CodeBlock title="Tool / Model Response" value={selectedEvent.response ?? selectedEvent.payload} />
                <CodeBlock title="Response Schema" value={endpoint?.response_schema ?? {}} />
              </Stack>
            )}

            {detailTab === 3 && (
              <Stack spacing={1.25} className="detailBody">
                <MiniTraceGraph
                  conversation={conversation}
                  events={traceEvents}
                  selectedEvent={selectedEvent}
                  onSelectEvent={(event) => {
                    onSelectEvent(event);
                    onDetailTabChange(3);
                  }}
                />
                <CodeBlock
                  title="Conversation Metadata"
                  value={{
                    pattern: conversation.metadata?.pattern,
                    tools_used: conversation.metadata?.tools_used,
                    domains: conversation.metadata?.category_domains,
                    planner_scenario: conversation.metadata?.planner_scenario,
                  }}
                />
              </Stack>
            )}
          </>
        ) : (
          <EmptyState title="No event selected" body="Choose a trace row or tool-call chip." />
        )}
      </Box>
    </Paper>
  );
}

function TraceRow({
  event,
  selected,
  hovered,
  onClick,
  onMouseEnter,
  onMouseLeave,
}: {
  event: TraceEvent;
  selected: boolean;
  hovered: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}) {
  return (
    <button
      type="button"
      className={`traceRow ${selected ? 'traceRow-selected' : ''} ${hovered ? 'traceRow-hovered' : ''}`}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <Box className="traceRail">
        <Box className="traceDot" sx={{ backgroundColor: eventColors[event.status] }} />
        <Box className="traceLine" />
      </Box>
      <Box className="traceRowContent">
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
          <Stack direction="row" alignItems="center" spacing={0.75} className="traceLabelGroup">
            <EventGlyph kind={event.kind} />
            <Typography className="traceLabel">{event.label}</Typography>
          </Stack>
          <Typography className="traceTurn">#{event.order}</Typography>
        </Stack>
        <Typography className="traceSubtitle">{event.subtitle}</Typography>
        {event.endpoint && (
          <Typography className="traceEndpoint">{event.endpoint}</Typography>
        )}
      </Box>
    </button>
  );
}

function EventGlyph({ kind }: { kind: TraceKind }) {
  if (kind === 'tool_call') {
    return <DataObject fontSize="small" color="primary" />;
  }
  if (kind === 'tool_response') {
    return <Memory fontSize="small" color="secondary" />;
  }
  if (kind === 'judge') {
    return <Insights fontSize="small" color="warning" />;
  }
  if (kind === 'repair') {
    return <BugReport fontSize="small" color="error" />;
  }
  if (kind === 'assistant') {
    return <AccountTree fontSize="small" color="secondary" />;
  }
  return <ChatBubbleOutline fontSize="small" color="primary" />;
}

function StatusDot({ status }: { status: TraceStatus }) {
  return (
    <Tooltip title={status}>
      <Box className="statusDot" sx={{ backgroundColor: eventColors[status] }} />
    </Tooltip>
  );
}

function MiniTraceGraph({
  conversation,
  events,
  selectedEvent,
  onSelectEvent,
}: {
  conversation: Conversation;
  events: TraceEvent[];
  selectedEvent: TraceEvent;
  onSelectEvent: (event: TraceEvent) => void;
}) {
  const [hoveredEventId, setHoveredEventId] = useState('');
  const path = events.filter((event) => event.kind === 'tool_call');
  const width = 760;
  const height = 250;
  const step = path.length > 1 ? (width - 120) / (path.length - 1) : 0;
  const selectedIndex = path.findIndex((event) => event.id === selectedEvent.id);
  const activeGraphEvent = path.find((event) => event.id === hoveredEventId) ??
    path.find((event) => event.id === selectedEvent.id) ??
    path[0];

  if (!path.length) {
    return <EmptyState title="No tool path" body="This chat has no tool-call events." />;
  }

  return (
    <Box className="miniGraph">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Tool-call trace graph">
        <rect width={width} height={height} fill="#ffffff" />
        {path.slice(1).map((event, index) => {
          const source = path[index];
          const x1 = 60 + index * step;
          const x2 = 60 + (index + 1) * step;
          const y = 118;
          const color = edgeColors[event.relation?.edge_type ?? 'io_chain'] ?? '#38bdf8';
          return (
            <g key={`${source.id}-${event.id}`}>
              <line x1={x1} y1={y} x2={x2} y2={y} stroke={color} strokeWidth="4" strokeOpacity="0.85" />
              <text x={(x1 + x2) / 2} y={y - 14} textAnchor="middle" className="miniGraphEdgeLabel">
                {event.relation?.edge_type ?? 'sequence'}
              </text>
            </g>
          );
        })}
        {path.map((event, index) => {
          const x = 60 + index * step;
          const selected = selectedIndex === index;
          const hovered = hoveredEventId === event.id;
          return (
            <g
              key={event.id}
              className="miniGraphNode"
              role="button"
              tabIndex={0}
              aria-label={`Select ${event.endpoint ?? event.label}`}
              onClick={() => onSelectEvent(event)}
              onKeyDown={(keyboardEvent) => {
                if (keyboardEvent.key === 'Enter' || keyboardEvent.key === ' ') {
                  keyboardEvent.preventDefault();
                  onSelectEvent(event);
                }
              }}
              onMouseEnter={() => setHoveredEventId(event.id)}
              onMouseLeave={() => setHoveredEventId('')}
            >
              <title>{event.endpoint}</title>
              <circle
                cx={x}
                cy={118}
                r={selected || hovered ? 19 : 14}
                fill={selected ? '#ed6c02' : hovered ? '#e3f2fd' : '#ffffff'}
                stroke={selected ? '#ed6c02' : '#1976d2'}
                strokeWidth={selected || hovered ? 4 : 2}
              />
              <text
                x={x}
                y={124}
                textAnchor="middle"
                className={selected ? 'miniGraphIndex miniGraphIndex-selected' : 'miniGraphIndex'}
              >
                {index + 1}
              </text>
              <text x={x} y={164} textAnchor="middle" className="miniGraphLabel">
                {shortEndpoint(event.endpoint ?? '')}
              </text>
            </g>
          );
        })}
      </svg>
      <Box className="graphSelection">
        <KeyValue label="Selected node" value={activeGraphEvent?.endpoint ?? 'none'} />
        <KeyValue
          label="Relation"
          value={activeGraphEvent?.relation?.edge_type ?? (activeGraphEvent ? 'sequence start' : 'none')}
        />
      </Box>
      <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap className="graphLegend">
        <Chip size="small" label={conversation.metadata?.pattern ?? 'unknown pattern'} color="primary" />
        {(conversation.metadata?.category_domains ?? []).map((domain) => (
          <Chip key={domain} size="small" label={domain} variant="outlined" />
        ))}
      </Stack>
    </Box>
  );
}

function ScorePill({ score }: { score: number }) {
  const color = score >= 4.5 ? 'success' : score >= 3.5 ? 'warning' : 'error';
  return <Chip size="small" color={color} label={formatNumber(score, 2)} className="scorePill" />;
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <Box className="keyValue">
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2">{value}</Typography>
    </Box>
  );
}

function CodeBlock({ title, value }: { title: string; value: unknown }) {
  return (
    <Box>
      <Typography variant="caption" color="text.secondary">
        {title}
      </Typography>
      <pre className="codeBlock">{JSON.stringify(value, null, 2)}</pre>
    </Box>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <Box className="emptyState">
      <InfoOutlined color="action" />
      <Typography variant="h3">{title}</Typography>
      <Typography color="text.secondary">{body}</Typography>
    </Box>
  );
}

function normalizeBundle(bundle: DashboardBundle): DashboardBundle {
  return {
    ...emptyBundle,
    ...bundle,
    conversations: bundle.conversations ?? [],
    liveSamples: bundle.liveSamples ?? [],
    runA: bundle.runA ?? [],
    runB: bundle.runB ?? [],
    artifacts: {
      ...emptyBundle.artifacts,
      ...(bundle.artifacts ?? {}),
      endpoints: bundle.artifacts?.endpoints ?? [],
      edges: bundle.artifacts?.edges ?? [],
    },
  };
}

function parseJsonl(text: string): Conversation[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Conversation);
}

function filterConversations(
  conversations: Conversation[],
  query: string,
  selectedTools: string[],
  toolMatchMode: 'any' | 'all',
) {
  const needle = query.trim().toLowerCase();
  return conversations.filter((conversation) => {
    const haystack = [
      conversation.conversation_id,
      conversation.metadata?.model,
      conversation.metadata?.pattern,
      conversation.metadata?.planner_scenario?.scenario,
      ...(conversation.metadata?.tools_used ?? []),
      ...(conversation.metadata?.category_domains ?? []),
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    const conversationTools = conversation.metadata?.tools_used ?? [];
    const matchesSearch = !needle || haystack.includes(needle);
    const matchesTools =
      selectedTools.length === 0 ||
      (toolMatchMode === 'all'
        ? selectedTools.every((tool) => conversationTools.includes(tool))
        : selectedTools.some((tool) => conversationTools.includes(tool)));
    return matchesSearch && matchesTools;
  });
}

function buildTraceEvents(conversation: Conversation, edgeMap: Map<string, Edge>): TraceEvent[] {
  const events: TraceEvent[] = [];
  const priorToolValues: string[] = [];
  let previousEndpoint: string | undefined;
  let toolTraceIndex = 0;
  let order = 1;

  conversation.messages.forEach((message, turnIndex) => {
    if (message.role === 'user') {
      events.push({
        id: `${conversation.conversation_id}-${turnIndex}-user`,
        turnIndex,
        order: order++,
        kind: 'user',
        status: 'info',
        label: 'user message',
        subtitle: truncate(formatContent(message.content), 88),
        message,
        payload: { role: message.role, content: message.content },
      });
      return;
    }

    if (message.role === 'assistant') {
      const content = formatContent(message.content);
      if (content) {
        events.push({
          id: `${conversation.conversation_id}-${turnIndex}-assistant`,
          turnIndex,
          order: order++,
          kind: 'assistant',
          status: 'ok',
          label: 'assistant response',
          subtitle: truncate(content, 88),
          message,
          payload: { role: message.role, content: message.content },
        });
      }

      const followingTools = conversation.messages
        .slice(turnIndex + 1)
        .filter((candidate) => candidate.role === 'tool');

      (message.tool_calls ?? []).forEach((call, callIndex) => {
        const argValues = flattenPrimitiveValues(call.arguments);
        const groundedArgs = argValues.filter((value) => priorToolValues.includes(value));
        const response = followingTools[callIndex]?.content ?? followingTools[0]?.content ?? null;
        const relation = previousEndpoint ? edgeMap.get(edgeKey(previousEndpoint, call.endpoint)) : undefined;
        const stepTrace =
          conversation.step_trace?.[toolTraceIndex] ??
          conversation.step_trace?.find((step) => step.endpoint === call.endpoint);
        toolTraceIndex += 1;
        events.push({
          id: `${conversation.conversation_id}-${turnIndex}-tool-call-${callIndex}`,
          turnIndex,
          order: order++,
          kind: 'tool_call',
          status: relation || !previousEndpoint ? 'ok' : 'warn',
          label: `function_call: ${shortEndpoint(call.endpoint)}`,
          subtitle: relation
            ? `${relation.edge_type} from ${shortEndpoint(previousEndpoint ?? '')}`
            : previousEndpoint
              ? 'no graph edge from previous tool'
              : 'first tool call in trace',
          endpoint: call.endpoint,
          args: call.arguments ?? {},
          response,
          previousEndpoint,
          relation,
          groundedArgs,
          stepTrace,
          payload: {
            role: message.role,
            endpoint: call.endpoint,
            arguments: call.arguments,
            grounded_args: groundedArgs,
            previous_endpoint: previousEndpoint,
            graph_relation: relation,
            step_trace: stepTrace,
          },
        });
        previousEndpoint = call.endpoint;
      });
      return;
    }

    priorToolValues.push(...flattenPrimitiveValues(message.content));
    events.push({
      id: `${conversation.conversation_id}-${turnIndex}-tool-response`,
      turnIndex,
      order: order++,
      kind: 'tool_response',
      status: 'ok',
      label: 'function_response',
      subtitle: truncate(formatContent(message.content), 88),
      endpoint: previousEndpoint,
      response: message.content,
      payload: { role: message.role, endpoint: previousEndpoint, content: message.content },
    });
  });

  const score = overallScore(conversation);
  events.push({
    id: `${conversation.conversation_id}-judge`,
    turnIndex: Math.max(0, conversation.messages.length - 1),
    order: order++,
    kind: 'judge',
    status: score >= 4.5 ? 'ok' : score >= 3.5 ? 'warn' : 'error',
    label: 'judge scores',
    subtitle: `overall ${formatNumber(score, 2)} | correctness ${formatNumber(scoreDimension(conversation, 'tool_correctness'), 2)}`,
    score,
    payload: conversation.judge_scores ?? {},
  });

  const repairs = conversation.metadata?.repair_attempts ?? 0;
  if (repairs > 0) {
    events.push({
      id: `${conversation.conversation_id}-repair`,
      turnIndex: Math.max(0, conversation.messages.length - 1),
      order: order++,
      kind: 'repair',
      status: 'warn',
      label: 'repair attempts',
      subtitle: `${repairs} repair attempt${repairs === 1 ? '' : 's'} recorded`,
      payload: {
        repair_attempts: repairs,
        threshold_related_score: score,
      },
    });
  }

  return events;
}

function overallScore(conversation: Conversation): number {
  const overall = conversation.judge_scores?.overall;
  if (typeof overall === 'number') {
    return overall;
  }
  if (typeof overall?.score === 'number') {
    return overall.score;
  }
  return 0;
}

function scoreDimension(conversation: Conversation, key: keyof JudgeScores): number {
  const value = conversation.judge_scores?.[key];
  if (typeof value === 'number') {
    return value;
  }
  if (typeof value?.score === 'number') {
    return value.score;
  }
  return 0;
}

function computeMetrics(conversations: Conversation[]) {
  const totalConversations = conversations.length;
  const totalToolCalls = conversations.reduce(
    (sum, conversation) => sum + (conversation.metadata?.num_tool_calls ?? countToolCalls(conversation)),
    0,
  );
  const scores = conversations.map(overallScore).filter((score) => score > 0);
  return {
    totalConversations,
    totalToolCalls,
    meanScore: average(scores),
    reviewCount: conversations.filter(
      (conversation) => overallScore(conversation) < 4.5 || (conversation.metadata?.repair_attempts ?? 0) > 0,
    ).length,
  };
}

function countToolCalls(conversation: Conversation) {
  return conversation.messages.reduce((sum, message) => sum + (message.tool_calls?.length ?? 0), 0);
}

function collectToolOptions(conversations: Conversation[]) {
  const counts = conversations
    .flatMap((conversation) => conversation.metadata?.tools_used ?? [])
    .reduce<Record<string, number>>((acc, tool) => {
      acc[tool] = (acc[tool] ?? 0) + 1;
      return acc;
    }, {});
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([tool]) => tool);
}

function flattenPrimitiveValues(value: unknown): string[] {
  if (value === null || value === undefined) {
    return [];
  }
  if (['string', 'number', 'boolean'].includes(typeof value)) {
    return [String(value)];
  }
  if (Array.isArray(value)) {
    return value.flatMap(flattenPrimitiveValues);
  }
  if (typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).flatMap(flattenPrimitiveValues);
  }
  return [];
}

function formatContent(content: Message['content']) {
  if (content === null || content === undefined) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  return JSON.stringify(content, null, 2);
}

function formatNumber(value: number, digits: number) {
  if (!Number.isFinite(value)) {
    return '0';
  }
  return value.toFixed(digits);
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function edgeKey(source: string, target: string) {
  return `${source}->${target}`;
}

function shortEndpoint(endpoint: string) {
  const parts = endpoint.split('/');
  return parts[1] ?? endpoint;
}

function truncate(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 3)}...`;
}

export default App;
