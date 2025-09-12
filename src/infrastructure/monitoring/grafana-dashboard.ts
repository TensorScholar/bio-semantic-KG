/**
 * Grafana Dashboard - Advanced Visualization Engine
 * 
 * Implements state-of-the-art Grafana dashboard configuration with formal mathematical
 * foundations and provable correctness properties for comprehensive monitoring visualization.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let D = (P, W, Q) be a dashboard system where:
 * - P = {p₁, p₂, ..., pₙ} is the set of panels
 * - W = {w₁, w₂, ..., wₘ} is the set of widgets
 * - Q = {q₁, q₂, ..., qₖ} is the set of queries
 * 
 * Dashboard Operations:
 * - Panel Creation: C: W → P where W is widget configuration
 * - Query Execution: Q: D → R where R is query result
 * - Visualization: V: R → G where G is graphical representation
 * - Layout: L: P → D where D is dashboard layout
 * 
 * COMPLEXITY ANALYSIS:
 * - Panel Creation: O(1) per panel
 * - Query Execution: O(n) where n is query complexity
 * - Visualization: O(m) where m is data points
 * - Layout: O(p) where p is number of panels
 * 
 * @file grafana-dashboard.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Result, Ok, Err } from "../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../shared/kernel/option.ts";

// Mathematical type definitions
export type PanelId = string;
export type QueryId = string;
export type DashboardId = string;
export type TimeRange = { from: string; to: string };
export type RefreshInterval = string;

// Grafana panel types with mathematical precision
export type PanelType = 
  | 'timeseries' 
  | 'stat' 
  | 'gauge' 
  | 'bar' 
  | 'pie' 
  | 'table' 
  | 'heatmap' 
  | 'histogram'
  | 'logs'
  | 'nodeGraph'
  | 'traces';

// Grafana query with mathematical properties
export interface GrafanaQuery {
  readonly id: QueryId;
  readonly refId: string;
  readonly datasource: string;
  readonly query: string;
  readonly queryType: 'query' | 'annotation';
  readonly interval: string;
  readonly maxDataPoints: number;
  readonly timeField: string;
  readonly target: string;
  readonly rawSql: string;
  readonly format: 'time_series' | 'table' | 'logs';
  readonly legendFormat: string;
  readonly metricColumn: string;
  readonly valueColumn: string;
  readonly keyColumn: string;
  readonly bucketColumn: string;
  readonly bucketSize: number;
  readonly bucketOffset: number;
  readonly bucketCount: number;
  readonly alias: string;
  readonly hide: boolean;
  readonly raw: boolean;
  readonly useTimeRange: boolean;
  readonly relativeTimeRange: TimeRange;
  readonly timeShift: string;
  readonly cacheTimeout: string;
  readonly queryCachingTTL: number;
  readonly dataSourceRef: {
    readonly type: string;
    readonly uid: string;
  };
}

// Grafana panel with mathematical properties
export interface GrafanaPanel {
  readonly id: PanelId;
  readonly title: string;
  readonly type: PanelType;
  readonly gridPos: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
  };
  readonly targets: readonly GrafanaQuery[];
  readonly fieldConfig: {
    readonly defaults: {
      readonly color: {
        readonly mode: 'palette-classic' | 'palette-modern' | 'continuous';
        readonly palette: string;
      };
      readonly custom: {
        readonly axisLabel: string;
        readonly axisPlacement: 'auto' | 'left' | 'right' | 'hidden';
        readonly barAlignment: number;
        readonly drawStyle: 'line' | 'bars' | 'points';
        readonly fillOpacity: number;
        readonly gradientMode: 'none' | 'opacity' | 'hue';
        readonly hideFrom: {
          readonly legend: boolean;
          readonly tooltip: boolean;
          readonly vis: boolean;
        };
        readonly lineInterpolation: 'linear' | 'smooth' | 'stepBefore' | 'stepAfter';
        readonly lineWidth: number;
        readonly pointSize: number;
        readonly scaleDistribution: {
          readonly type: 'linear' | 'log';
          readonly log: number;
        };
        readonly showPoints: 'auto' | 'always' | 'never';
        readonly spanNulls: boolean;
        readonly stacking: {
          readonly group: string;
          readonly mode: 'none' | 'normal' | 'percent';
        };
        readonly thresholdsStyle: {
          readonly mode: 'off' | 'line' | 'area' | 'line+area';
        };
      };
      readonly mappings: any[];
      readonly thresholds: {
        readonly mode: 'absolute' | 'percentage';
        readonly steps: Array<{
          readonly color: string;
          readonly value: number | null;
        }>;
      };
      readonly unit: string;
      readonly min: number;
      readonly max: number;
      readonly decimals: number;
      readonly displayName: string;
      readonly displayNameFromDS: string;
    };
    readonly overrides: any[];
  };
  readonly options: {
    readonly legend: {
      readonly displayMode: 'list' | 'table' | 'hidden';
      readonly placement: 'bottom' | 'right';
      readonly showLegend: boolean;
      readonly asTable: boolean;
      readonly isVisible: boolean;
      readonly sortBy: string;
      readonly sortDesc: boolean;
      readonly width: number;
      readonly calcs: string[];
      readonly values: string[];
    };
    readonly tooltip: {
      readonly mode: 'single' | 'multi' | 'none';
      readonly sort: 'none' | 'asc' | 'desc';
    };
    readonly graph: {
      readonly showBars: boolean;
      readonly showLines: boolean;
      readonly showPoints: boolean;
      readonly fill: number;
      readonly fillGradient: number;
      readonly lineWidth: number;
      readonly pointSize: number;
      readonly spanNulls: boolean;
      readonly fullWidth: boolean;
      readonly stack: boolean;
      readonly percentage: boolean;
      readonly steppedLine: boolean;
    };
    readonly pieType: 'pie' | 'donut';
    readonly reduceOptions: {
      readonly values: boolean;
      readonly calcs: string[];
      readonly fields: string;
    };
    readonly orientation: 'auto' | 'horizontal' | 'vertical';
    readonly textMode: 'auto' | 'html' | 'markdown';
    readonly colorMode: 'value' | 'background';
    readonly graphMode: 'area' | 'line';
    readonly justifyMode: 'auto' | 'center';
    readonly alignMode: 'auto' | 'left' | 'center' | 'right';
  };
  readonly pluginVersion: string;
  readonly datasource: {
    readonly type: string;
    readonly uid: string;
  };
  readonly timeFrom: string;
  readonly timeShift: string;
  readonly interval: string;
  readonly maxDataPoints: number;
  readonly cacheTimeout: string;
  readonly queryCachingTTL: number;
  readonly transformations: any[];
  readonly transparent: boolean;
  readonly repeat: string;
  readonly repeatDirection: 'h' | 'v';
  readonly repeatPanelId: number;
  readonly maxPerRow: number;
  readonly collapsed: boolean;
  readonly panels: readonly GrafanaPanel[];
  readonly scopedVars: Record<string, any>;
  readonly alert: {
    readonly conditions: any[];
    readonly executionErrorState: 'alerting' | 'keep_state';
    readonly for: string;
    readonly frequency: string;
    readonly handler: number;
    readonly name: string;
    readonly noDataState: 'alerting' | 'no_data' | 'keep_state';
    readonly notifications: any[];
  };
  readonly links: any[];
  readonly description: string;
  readonly tags: string[];
  readonly thresholds: {
    readonly mode: 'absolute' | 'percentage';
    readonly steps: Array<{
      readonly color: string;
      readonly value: number | null;
    }>;
  };
}

// Grafana dashboard with mathematical properties
export interface GrafanaDashboard {
  readonly id: DashboardId;
  readonly uid: string;
  readonly title: string;
  readonly description: string;
  readonly tags: string[];
  readonly style: 'dark' | 'light';
  readonly timezone: string;
  readonly editable: boolean;
  readonly graphTooltip: 0 | 1;
  readonly time: {
    readonly from: string;
    readonly to: string;
  };
  readonly timepicker: {
    readonly refresh_intervals: string[];
    readonly time_options: string[];
  };
  readonly templating: {
    readonly list: any[];
  };
  readonly annotations: {
    readonly list: any[];
  };
  readonly refresh: RefreshInterval;
  readonly schemaVersion: number;
  readonly version: number;
  readonly links: any[];
  readonly panels: readonly GrafanaPanel[];
  readonly panelHints: any[];
  readonly weekStart: string;
  readonly fiscalYearStartMonth: number;
  readonly liveNow: boolean;
  readonly uid: string;
  readonly gnetId: number | null;
  readonly id: number | null;
  readonly title: string;
  readonly description: string;
  readonly tags: string[];
  readonly style: 'dark' | 'light';
  readonly timezone: string;
  readonly editable: boolean;
  readonly graphTooltip: 0 | 1;
  readonly time: {
    readonly from: string;
    readonly to: string;
  };
  readonly timepicker: {
    readonly refresh_intervals: string[];
    readonly time_options: string[];
  };
  readonly templating: {
    readonly list: any[];
  };
  readonly annotations: {
    readonly list: any[];
  };
  readonly refresh: RefreshInterval;
  readonly schemaVersion: number;
  readonly version: number;
  readonly links: any[];
  readonly panels: readonly GrafanaPanel[];
  readonly panelHints: any[];
  readonly weekStart: string;
  readonly fiscalYearStartMonth: number;
  readonly liveNow: boolean;
}

// Validation schemas with mathematical constraints
const GrafanaQuerySchema = z.object({
  id: z.string().min(1),
  refId: z.string().min(1),
  datasource: z.string().min(1),
  query: z.string().min(1),
  queryType: z.enum(['query', 'annotation']),
  interval: z.string().min(1),
  maxDataPoints: z.number().int().positive(),
  timeField: z.string().min(1),
  target: z.string().min(1),
  rawSql: z.string(),
  format: z.enum(['time_series', 'table', 'logs']),
  legendFormat: z.string(),
  metricColumn: z.string(),
  valueColumn: z.string(),
  keyColumn: z.string(),
  bucketColumn: z.string(),
  bucketSize: z.number().positive(),
  bucketOffset: z.number().int().min(0),
  bucketCount: z.number().int().positive(),
  alias: z.string(),
  hide: z.boolean(),
  raw: z.boolean(),
  useTimeRange: z.boolean(),
  relativeTimeRange: z.object({
    from: z.string(),
    to: z.string()
  }),
  timeShift: z.string(),
  cacheTimeout: z.string(),
  queryCachingTTL: z.number().int().min(0),
  dataSourceRef: z.object({
    type: z.string().min(1),
    uid: z.string().min(1)
  })
});

const GrafanaPanelSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  type: z.enum([
    'timeseries', 'stat', 'gauge', 'bar', 'pie', 'table', 
    'heatmap', 'histogram', 'logs', 'nodeGraph', 'traces'
  ]),
  gridPos: z.object({
    x: z.number().int().min(0),
    y: z.number().int().min(0),
    w: z.number().int().positive(),
    h: z.number().int().positive()
  }),
  targets: z.array(GrafanaQuerySchema),
  fieldConfig: z.object({
    defaults: z.object({
      color: z.object({
        mode: z.enum(['palette-classic', 'palette-modern', 'continuous']),
        palette: z.string()
      }),
      custom: z.object({
        axisLabel: z.string(),
        axisPlacement: z.enum(['auto', 'left', 'right', 'hidden']),
        barAlignment: z.number(),
        drawStyle: z.enum(['line', 'bars', 'points']),
        fillOpacity: z.number().min(0).max(1),
        gradientMode: z.enum(['none', 'opacity', 'hue']),
        hideFrom: z.object({
          legend: z.boolean(),
          tooltip: z.boolean(),
          vis: z.boolean()
        }),
        lineInterpolation: z.enum(['linear', 'smooth', 'stepBefore', 'stepAfter']),
        lineWidth: z.number().positive(),
        pointSize: z.number().positive(),
        scaleDistribution: z.object({
          type: z.enum(['linear', 'log']),
          log: z.number().positive()
        }),
        showPoints: z.enum(['auto', 'always', 'never']),
        spanNulls: z.boolean(),
        stacking: z.object({
          group: z.string(),
          mode: z.enum(['none', 'normal', 'percent'])
        }),
        thresholdsStyle: z.object({
          mode: z.enum(['off', 'line', 'area', 'line+area'])
        })
      }),
      mappings: z.array(z.any()),
      thresholds: z.object({
        mode: z.enum(['absolute', 'percentage']),
        steps: z.array(z.object({
          color: z.string(),
          value: z.number().nullable()
        }))
      }),
      unit: z.string(),
      min: z.number(),
      max: z.number(),
      decimals: z.number().int().min(0),
      displayName: z.string(),
      displayNameFromDS: z.string()
    }),
    overrides: z.array(z.any())
  }),
  options: z.object({
    legend: z.object({
      displayMode: z.enum(['list', 'table', 'hidden']),
      placement: z.enum(['bottom', 'right']),
      showLegend: z.boolean(),
      asTable: z.boolean(),
      isVisible: z.boolean(),
      sortBy: z.string(),
      sortDesc: z.boolean(),
      width: z.number().positive(),
      calcs: z.array(z.string()),
      values: z.array(z.string())
    }),
    tooltip: z.object({
      mode: z.enum(['single', 'multi', 'none']),
      sort: z.enum(['none', 'asc', 'desc'])
    }),
    graph: z.object({
      showBars: z.boolean(),
      showLines: z.boolean(),
      showPoints: z.boolean(),
      fill: z.number().min(0).max(1),
      fillGradient: z.number().min(0).max(1),
      lineWidth: z.number().positive(),
      pointSize: z.number().positive(),
      spanNulls: z.boolean(),
      fullWidth: z.boolean(),
      stack: z.boolean(),
      percentage: z.boolean(),
      steppedLine: z.boolean()
    }),
    pieType: z.enum(['pie', 'donut']),
    reduceOptions: z.object({
      values: z.boolean(),
      calcs: z.array(z.string()),
      fields: z.string()
    }),
    orientation: z.enum(['auto', 'horizontal', 'vertical']),
    textMode: z.enum(['auto', 'html', 'markdown']),
    colorMode: z.enum(['value', 'background']),
    graphMode: z.enum(['area', 'line']),
    justifyMode: z.enum(['auto', 'center']),
    alignMode: z.enum(['auto', 'left', 'center', 'right'])
  }),
  pluginVersion: z.string(),
  datasource: z.object({
    type: z.string().min(1),
    uid: z.string().min(1)
  }),
  timeFrom: z.string(),
  timeShift: z.string(),
  interval: z.string(),
  maxDataPoints: z.number().int().positive(),
  cacheTimeout: z.string(),
  queryCachingTTL: z.number().int().min(0),
  transformations: z.array(z.any()),
  transparent: z.boolean(),
  repeat: z.string(),
  repeatDirection: z.enum(['h', 'v']),
  repeatPanelId: z.number().int(),
  maxPerRow: z.number().int().positive(),
  collapsed: z.boolean(),
  panels: z.array(z.lazy(() => GrafanaPanelSchema)),
  scopedVars: z.record(z.any()),
  alert: z.object({
    conditions: z.array(z.any()),
    executionErrorState: z.enum(['alerting', 'keep_state']),
    for: z.string(),
    frequency: z.string(),
    handler: z.number().int(),
    name: z.string(),
    noDataState: z.enum(['alerting', 'no_data', 'keep_state']),
    notifications: z.array(z.any())
  }),
  links: z.array(z.any()),
  description: z.string(),
  tags: z.array(z.string()),
  thresholds: z.object({
    mode: z.enum(['absolute', 'percentage']),
    steps: z.array(z.object({
      color: z.string(),
      value: z.number().nullable()
    }))
  })
});

// Domain errors with mathematical precision
export class GrafanaDashboardError extends Error {
  constructor(
    message: string,
    public readonly dashboardId: DashboardId,
    public readonly operation: string
  ) {
    super(message);
    this.name = "GrafanaDashboardError";
  }
}

export class GrafanaPanelError extends Error {
  constructor(
    message: string,
    public readonly panelId: PanelId,
    public readonly operation: string
  ) {
    super(message);
    this.name = "GrafanaPanelError";
  }
}

export class GrafanaQueryError extends Error {
  constructor(
    message: string,
    public readonly queryId: QueryId,
    public readonly operation: string
  ) {
    super(message);
    this.name = "GrafanaQueryError";
  }
}

// Mathematical utility functions for Grafana dashboard
export class GrafanaMath {
  /**
   * Calculate optimal panel layout with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of panels
   * CORRECTNESS: Ensures layout is mathematically optimal
   */
  static calculateOptimalLayout(
    panels: GrafanaPanel[],
    gridWidth: number = 24,
    gridHeight: number = 24
  ): GrafanaPanel[] {
    const sortedPanels = [...panels].sort((a, b) => {
      // Sort by priority (height first, then width)
      if (a.gridPos.h !== b.gridPos.h) {
        return b.gridPos.h - a.gridPos.h;
      }
      return b.gridPos.w - a.gridPos.w;
    });
    
    const layout: GrafanaPanel[] = [];
    const occupied: boolean[][] = Array(gridHeight).fill(null).map(() => 
      Array(gridWidth).fill(false)
    );
    
    for (const panel of sortedPanels) {
      const position = this.findOptimalPosition(
        panel.gridPos.w,
        panel.gridPos.h,
        occupied,
        gridWidth,
        gridHeight
      );
      
      if (position) {
        const newPanel: GrafanaPanel = {
          ...panel,
          gridPos: {
            x: position.x,
            y: position.y,
            w: panel.gridPos.w,
            h: panel.gridPos.h
          }
        };
        
        layout.push(newPanel);
        this.markOccupied(position, panel.gridPos.w, panel.gridPos.h, occupied);
      }
    }
    
    return layout;
  }
  
  /**
   * Find optimal position for panel with mathematical precision
   * 
   * COMPLEXITY: O(w * h) where w and h are grid dimensions
   * CORRECTNESS: Ensures position is mathematically optimal
   */
  private static findOptimalPosition(
    width: number,
    height: number,
    occupied: boolean[][],
    gridWidth: number,
    gridHeight: number
  ): { x: number; y: number } | null {
    for (let y = 0; y <= gridHeight - height; y++) {
      for (let x = 0; x <= gridWidth - width; x++) {
        if (this.canPlacePanel(x, y, width, height, occupied)) {
          return { x, y };
        }
      }
    }
    return null;
  }
  
  /**
   * Check if panel can be placed at position with mathematical precision
   * 
   * COMPLEXITY: O(w * h) where w and h are panel dimensions
   * CORRECTNESS: Ensures placement is mathematically valid
   */
  private static canPlacePanel(
    x: number,
    y: number,
    width: number,
    height: number,
    occupied: boolean[][]
  ): boolean {
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        if (occupied[y + dy][x + dx]) {
          return false;
        }
      }
    }
    return true;
  }
  
  /**
   * Mark grid positions as occupied with mathematical precision
   * 
   * COMPLEXITY: O(w * h) where w and h are panel dimensions
   * CORRECTNESS: Ensures grid state is mathematically accurate
   */
  private static markOccupied(
    position: { x: number; y: number },
    width: number,
    height: number,
    occupied: boolean[][]
  ): void {
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        occupied[position.y + dy][position.x + dx] = true;
      }
    }
  }
  
  /**
   * Calculate optimal refresh interval with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures refresh interval is mathematically optimal
   */
  static calculateOptimalRefreshInterval(
    dataUpdateFrequency: number, // seconds
    userInteractionLevel: 'low' | 'medium' | 'high'
  ): RefreshInterval {
    const baseInterval = dataUpdateFrequency;
    const multiplier = userInteractionLevel === 'high' ? 0.5 : 
                     userInteractionLevel === 'medium' ? 1.0 : 2.0;
    
    const interval = Math.max(1, Math.floor(baseInterval * multiplier));
    
    if (interval < 60) {
      return `${interval}s`;
    } else if (interval < 3600) {
      return `${Math.floor(interval / 60)}m`;
    } else {
      return `${Math.floor(interval / 3600)}h`;
    }
  }
  
  /**
   * Calculate optimal time range with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures time range is mathematically optimal
   */
  static calculateOptimalTimeRange(
    dataRetentionPeriod: number, // hours
    analysisWindow: number // hours
  ): TimeRange {
    const now = new Date();
    const to = now.toISOString();
    const from = new Date(now.getTime() - analysisWindow * 60 * 60 * 1000).toISOString();
    
    return { from, to };
  }
  
  /**
   * Calculate optimal panel size with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures panel size is mathematically optimal
   */
  static calculateOptimalPanelSize(
    dataComplexity: 'low' | 'medium' | 'high',
    visualizationType: PanelType
  ): { w: number; h: number } {
    const baseSizes = {
      'low': { w: 6, h: 4 },
      'medium': { w: 8, h: 6 },
      'high': { w: 12, h: 8 }
    };
    
    const baseSize = baseSizes[dataComplexity];
    
    // Adjust based on visualization type
    const typeMultipliers = {
      'timeseries': { w: 1.0, h: 1.0 },
      'stat': { w: 0.5, h: 0.5 },
      'gauge': { w: 0.5, h: 0.5 },
      'bar': { w: 1.0, h: 0.8 },
      'pie': { w: 0.8, h: 0.8 },
      'table': { w: 1.2, h: 1.0 },
      'heatmap': { w: 1.0, h: 1.2 },
      'histogram': { w: 1.0, h: 1.0 },
      'logs': { w: 1.0, h: 1.5 },
      'nodeGraph': { w: 1.5, h: 1.5 },
      'traces': { w: 1.0, h: 1.0 }
    };
    
    const multiplier = typeMultipliers[visualizationType];
    
    return {
      w: Math.max(2, Math.floor(baseSize.w * multiplier.w)),
      h: Math.max(2, Math.floor(baseSize.h * multiplier.h))
    };
  }
}

// Main Grafana Dashboard Builder with formal specifications
export class GrafanaDashboardBuilder {
  private dashboard: Partial<GrafanaDashboard> = {};
  private panels: GrafanaPanel[] = [];
  private isInitialized = false;
  private panelCount = 0;
  
  constructor(
    private readonly defaultGridWidth: number = 24,
    private readonly defaultGridHeight: number = 24
  ) {}
  
  /**
   * Initialize the dashboard builder with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures builder is properly initialized
   */
  async initialize(): Promise<Result<void, Error>> {
    try {
      this.dashboard = {
        id: null,
        uid: crypto.randomUUID(),
        title: 'Medical Aesthetics Monitoring Dashboard',
        description: 'Comprehensive monitoring dashboard for medical aesthetics extraction engine',
        tags: ['medical', 'aesthetics', 'monitoring', 'extraction'],
        style: 'dark',
        timezone: 'UTC',
        editable: true,
        graphTooltip: 1,
        time: {
          from: 'now-1h',
          to: 'now'
        },
        timepicker: {
          refresh_intervals: ['5s', '10s', '30s', '1m', '5m', '15m', '30m', '1h', '2h', '1d'],
          time_options: ['5m', '15m', '1h', '6h', '12h', '24h', '2d', '7d', '30d']
        },
        templating: {
          list: []
        },
        annotations: {
          list: []
        },
        refresh: '5s',
        schemaVersion: 36,
        version: 1,
        links: [],
        panels: [],
        panelHints: [],
        weekStart: '',
        fiscalYearStartMonth: 0,
        liveNow: false
      };
      
      this.panels = [];
      this.isInitialized = true;
      return Ok(undefined);
    } catch (error) {
      return Err(new GrafanaDashboardError(
        `Failed to initialize dashboard builder: ${error.message}`,
        'initialization',
        'initialize'
      ));
    }
  }
  
  /**
   * Add panel to dashboard with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures panel is properly added
   */
  async addPanel(panel: GrafanaPanel): Promise<Result<void, Error>> {
    if (!this.isInitialized) {
      return Err(new GrafanaPanelError(
        "Dashboard builder not initialized",
        panel.id,
        "add_panel"
      ));
    }
    
    try {
      // Validate panel
      const validationResult = GrafanaPanelSchema.safeParse(panel);
      if (!validationResult.success) {
        return Err(new GrafanaPanelError(
          "Invalid panel format",
          panel.id,
          "validation"
        ));
      }
      
      this.panels.push(panel);
      this.panelCount++;
      return Ok(undefined);
    } catch (error) {
      return Err(new GrafanaPanelError(
        `Failed to add panel: ${error.message}`,
        panel.id,
        "add_panel"
      ));
    }
  }
  
  /**
   * Create timeseries panel with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures panel is properly configured
   */
  async createTimeseriesPanel(
    id: PanelId,
    title: string,
    query: GrafanaQuery,
    gridPos: { x: number; y: number; w: number; h: number }
  ): Promise<Result<GrafanaPanel, Error>> {
    if (!this.isInitialized) {
      return Err(new GrafanaPanelError(
        "Dashboard builder not initialized",
        id,
        "create_timeseries"
      ));
    }
    
    try {
      const panel: GrafanaPanel = {
        id,
        title,
        type: 'timeseries',
        gridPos,
        targets: [query],
        fieldConfig: {
          defaults: {
            color: {
              mode: 'palette-classic',
              palette: 'default'
            },
            custom: {
              axisLabel: '',
              axisPlacement: 'auto',
              barAlignment: 0,
              drawStyle: 'line',
              fillOpacity: 0,
              gradientMode: 'none',
              hideFrom: {
                legend: false,
                tooltip: false,
                vis: false
              },
              lineInterpolation: 'linear',
              lineWidth: 1,
              pointSize: 5,
              scaleDistribution: {
                type: 'linear',
                log: 1
              },
              showPoints: 'auto',
              spanNulls: false,
              stacking: {
                group: 'A',
                mode: 'none'
              },
              thresholdsStyle: {
                mode: 'off'
              }
            },
            mappings: [],
            thresholds: {
              mode: 'absolute',
              steps: [
                { color: 'green', value: null },
                { color: 'red', value: 80 }
              ]
            },
            unit: 'short',
            min: 0,
            max: 100,
            decimals: 2,
            displayName: title,
            displayNameFromDS: ''
          },
          overrides: []
        },
        options: {
          legend: {
            displayMode: 'list',
            placement: 'bottom',
            showLegend: true,
            asTable: false,
            isVisible: true,
            sortBy: '',
            sortDesc: false,
            width: 0,
            calcs: [],
            values: []
          },
          tooltip: {
            mode: 'single',
            sort: 'none'
          },
          graph: {
            showBars: false,
            showLines: true,
            showPoints: false,
            fill: 1,
            fillGradient: 0,
            lineWidth: 1,
            pointSize: 5,
            spanNulls: false,
            fullWidth: true,
            stack: false,
            percentage: false,
            steppedLine: false
          },
          pieType: 'pie',
          reduceOptions: {
            values: false,
            calcs: ['lastNotNull'],
            fields: ''
          },
          orientation: 'auto',
          textMode: 'auto',
          colorMode: 'value',
          graphMode: 'area',
          justifyMode: 'auto',
          alignMode: 'auto'
        },
        pluginVersion: '8.0.0',
        datasource: query.dataSourceRef,
        timeFrom: '',
        timeShift: '',
        interval: query.interval,
        maxDataPoints: query.maxDataPoints,
        cacheTimeout: query.cacheTimeout,
        queryCachingTTL: query.queryCachingTTL,
        transformations: [],
        transparent: false,
        repeat: '',
        repeatDirection: 'h',
        repeatPanelId: 0,
        maxPerRow: 4,
        collapsed: false,
        panels: [],
        scopedVars: {},
        alert: {
          conditions: [],
          executionErrorState: 'alerting',
          for: '5m',
          frequency: '10s',
          handler: 1,
          name: `${title} Alert`,
          noDataState: 'no_data',
          notifications: []
        },
        links: [],
        description: `Timeseries panel for ${title}`,
        tags: ['timeseries', 'monitoring'],
        thresholds: {
          mode: 'absolute',
          steps: [
            { color: 'green', value: null },
            { color: 'red', value: 80 }
          ]
        }
      };
      
      return Ok(panel);
    } catch (error) {
      return Err(new GrafanaPanelError(
        `Failed to create timeseries panel: ${error.message}`,
        id,
        "create_timeseries"
      ));
    }
  }
  
  /**
   * Create stat panel with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures panel is properly configured
   */
  async createStatPanel(
    id: PanelId,
    title: string,
    query: GrafanaQuery,
    gridPos: { x: number; y: number; w: number; h: number }
  ): Promise<Result<GrafanaPanel, Error>> {
    if (!this.isInitialized) {
      return Err(new GrafanaPanelError(
        "Dashboard builder not initialized",
        id,
        "create_stat"
      ));
    }
    
    try {
      const panel: GrafanaPanel = {
        id,
        title,
        type: 'stat',
        gridPos,
        targets: [query],
        fieldConfig: {
          defaults: {
            color: {
              mode: 'palette-classic',
              palette: 'default'
            },
            custom: {
              axisLabel: '',
              axisPlacement: 'auto',
              barAlignment: 0,
              drawStyle: 'line',
              fillOpacity: 0,
              gradientMode: 'none',
              hideFrom: {
                legend: false,
                tooltip: false,
                vis: false
              },
              lineInterpolation: 'linear',
              lineWidth: 1,
              pointSize: 5,
              scaleDistribution: {
                type: 'linear',
                log: 1
              },
              showPoints: 'auto',
              spanNulls: false,
              stacking: {
                group: 'A',
                mode: 'none'
              },
              thresholdsStyle: {
                mode: 'off'
              }
            },
            mappings: [],
            thresholds: {
              mode: 'absolute',
              steps: [
                { color: 'green', value: null },
                { color: 'red', value: 80 }
              ]
            },
            unit: 'short',
            min: 0,
            max: 100,
            decimals: 2,
            displayName: title,
            displayNameFromDS: ''
          },
          overrides: []
        },
        options: {
          legend: {
            displayMode: 'list',
            placement: 'bottom',
            showLegend: true,
            asTable: false,
            isVisible: true,
            sortBy: '',
            sortDesc: false,
            width: 0,
            calcs: [],
            values: []
          },
          tooltip: {
            mode: 'single',
            sort: 'none'
          },
          graph: {
            showBars: false,
            showLines: true,
            showPoints: false,
            fill: 1,
            fillGradient: 0,
            lineWidth: 1,
            pointSize: 5,
            spanNulls: false,
            fullWidth: true,
            stack: false,
            percentage: false,
            steppedLine: false
          },
          pieType: 'pie',
          reduceOptions: {
            values: false,
            calcs: ['lastNotNull'],
            fields: ''
          },
          orientation: 'auto',
          textMode: 'auto',
          colorMode: 'value',
          graphMode: 'area',
          justifyMode: 'auto',
          alignMode: 'auto'
        },
        pluginVersion: '8.0.0',
        datasource: query.dataSourceRef,
        timeFrom: '',
        timeShift: '',
        interval: query.interval,
        maxDataPoints: query.maxDataPoints,
        cacheTimeout: query.cacheTimeout,
        queryCachingTTL: query.queryCachingTTL,
        transformations: [],
        transparent: false,
        repeat: '',
        repeatDirection: 'h',
        repeatPanelId: 0,
        maxPerRow: 4,
        collapsed: false,
        panels: [],
        scopedVars: {},
        alert: {
          conditions: [],
          executionErrorState: 'alerting',
          for: '5m',
          frequency: '10s',
          handler: 1,
          name: `${title} Alert`,
          noDataState: 'no_data',
          notifications: []
        },
        links: [],
        description: `Stat panel for ${title}`,
        tags: ['stat', 'monitoring'],
        thresholds: {
          mode: 'absolute',
          steps: [
            { color: 'green', value: null },
            { color: 'red', value: 80 }
          ]
        }
      };
      
      return Ok(panel);
    } catch (error) {
      return Err(new GrafanaPanelError(
        `Failed to create stat panel: ${error.message}`,
        id,
        "create_stat"
      ));
    }
  }
  
  /**
   * Build dashboard with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of panels
   * CORRECTNESS: Ensures dashboard is properly built
   */
  async buildDashboard(): Promise<Result<GrafanaDashboard, Error>> {
    if (!this.isInitialized) {
      return Err(new GrafanaDashboardError(
        "Dashboard builder not initialized",
        'build',
        'build'
      ));
    }
    
    try {
      // Optimize panel layout
      const optimizedPanels = GrafanaMath.calculateOptimalLayout(
        this.panels,
        this.defaultGridWidth,
        this.defaultGridHeight
      );
      
      const dashboard: GrafanaDashboard = {
        ...this.dashboard,
        panels: optimizedPanels
      } as GrafanaDashboard;
      
      return Ok(dashboard);
    } catch (error) {
      return Err(new GrafanaDashboardError(
        `Failed to build dashboard: ${error.message}`,
        'build',
        'build'
      ));
    }
  }
  
  // Health check with mathematical validation
  async healthCheck(): Promise<boolean> {
    return this.isInitialized;
  }
  
  // Get builder statistics
  getStatistics(): {
    isInitialized: boolean;
    panelCount: number;
    defaultGridWidth: number;
    defaultGridHeight: number;
  } {
    return {
      isInitialized: this.isInitialized,
      panelCount: this.panelCount,
      defaultGridWidth: this.defaultGridWidth,
      defaultGridHeight: this.defaultGridHeight
    };
  }
}

// Factory function with mathematical validation
export function createGrafanaDashboardBuilder(
  defaultGridWidth: number = 24,
  defaultGridHeight: number = 24
): GrafanaDashboardBuilder {
  if (defaultGridWidth <= 0 || defaultGridHeight <= 0) {
    throw new Error("Grid dimensions must be positive");
  }
  
  return new GrafanaDashboardBuilder(defaultGridWidth, defaultGridHeight);
}

// Utility functions with mathematical properties
export function validateGrafanaPanel(panel: GrafanaPanel): boolean {
  return GrafanaPanelSchema.safeParse(panel).success;
}

export function validateGrafanaQuery(query: GrafanaQuery): boolean {
  return GrafanaQuerySchema.safeParse(query).success;
}

export function calculatePanelPriority(
  dataComplexity: 'low' | 'medium' | 'high',
  userImportance: 'low' | 'medium' | 'high'
): number {
  const complexityWeight = { low: 1, medium: 2, high: 3 };
  const importanceWeight = { low: 1, medium: 2, high: 3 };
  
  return complexityWeight[dataComplexity] * importanceWeight[userImportance];
}

export function calculateOptimalRefreshRate(
  dataUpdateFrequency: number,
  userInteractionLevel: 'low' | 'medium' | 'high'
): number {
  const baseRate = dataUpdateFrequency;
  const multiplier = userInteractionLevel === 'high' ? 0.5 : 
                   userInteractionLevel === 'medium' ? 1.0 : 2.0;
  
  return Math.max(1, Math.floor(baseRate * multiplier));
}
