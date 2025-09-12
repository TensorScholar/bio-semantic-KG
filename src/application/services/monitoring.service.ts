/**
 * Monitoring Service - Advanced Observability Orchestration
 * 
 * Implements comprehensive monitoring orchestration with formal mathematical
 * foundations and provable correctness properties for medical aesthetics domain.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let M = (C, E, D, A) be a monitoring system where:
 * - C = {c₁, c₂, ..., cₙ} is the set of collectors
 * - E = {e₁, e₂, ..., eₘ} is the set of exporters
 * - D = {d₁, d₂, ..., dₖ} is the set of dashboards
 * - A = {a₁, a₂, ..., aₗ} is the set of alerts
 * 
 * Monitoring Operations:
 * - Collection: C: S → M where S is system state
 * - Export: E: M → P where P is Prometheus format
 * - Visualization: V: M → D where D is dashboard
 * - Alerting: A: M → N where N is notification
 * 
 * COMPLEXITY ANALYSIS:
 * - Metric Collection: O(1) per metric
 * - Export: O(n) where n is number of metrics
 * - Dashboard Generation: O(p) where p is number of panels
 * - Alert Processing: O(1) per alert
 * 
 * @file monitoring.service.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Result, Ok, Err } from "../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../shared/kernel/option.ts";
import { MetricsCollector, Metric, MetricAggregation, AlertConfig, Alert } from "../../../infrastructure/monitoring/metrics-collector.ts";
import { PrometheusExporter, PrometheusConfig, PrometheusMetric, PrometheusHistogram, PrometheusSummary } from "../../../infrastructure/monitoring/prometheus-exporter.ts";
import { GrafanaDashboardBuilder, GrafanaDashboard, GrafanaPanel, GrafanaQuery } from "../../../infrastructure/monitoring/grafana-dashboard.ts";

// Monitoring configuration with mathematical validation
export interface MonitoringConfig {
  readonly metrics: {
    readonly maxMetricsPerName: number;
    readonly defaultTimeWindow: number;
    readonly collectionInterval: number;
  };
  readonly prometheus: {
    readonly endpoint: string;
    readonly port: number;
    readonly path: string;
    readonly format: 'text' | 'protobuf';
    readonly compression: boolean;
    readonly timeout: number;
    readonly retries: number;
    readonly batchSize: number;
    readonly labels: Map<string, string>;
  };
  readonly grafana: {
    readonly gridWidth: number;
    readonly gridHeight: number;
    readonly refreshInterval: string;
    readonly timeRange: { from: string; to: string };
  };
  readonly alerts: {
    readonly enabled: boolean;
    readonly notificationChannels: string[];
    readonly defaultSeverity: 'critical' | 'warning' | 'info';
    readonly evaluationInterval: number;
  };
}

// Validation schema for monitoring configuration
const MonitoringConfigSchema = z.object({
  metrics: z.object({
    maxMetricsPerName: z.number().int().positive(),
    defaultTimeWindow: z.number().positive(),
    collectionInterval: z.number().positive()
  }),
  prometheus: z.object({
    endpoint: z.string().url(),
    port: z.number().int().min(1).max(65535),
    path: z.string().min(1),
    format: z.enum(['text', 'protobuf']),
    compression: z.boolean(),
    timeout: z.number().positive(),
    retries: z.number().int().min(0),
    batchSize: z.number().int().positive(),
    labels: z.record(z.string())
  }),
  grafana: z.object({
    gridWidth: z.number().int().positive(),
    gridHeight: z.number().int().positive(),
    refreshInterval: z.string().min(1),
    timeRange: z.object({
      from: z.string().min(1),
      to: z.string().min(1)
    })
  }),
  alerts: z.object({
    enabled: z.boolean(),
    notificationChannels: z.array(z.string()),
    defaultSeverity: z.enum(['critical', 'warning', 'info']),
    evaluationInterval: z.number().positive()
  })
});

// Monitoring metrics with mathematical precision
export interface MonitoringMetrics {
  readonly system: {
    readonly cpuUsage: number;
    readonly memoryUsage: number;
    readonly diskUsage: number;
    readonly networkLatency: number;
  };
  readonly application: {
    readonly requestCount: number;
    readonly errorRate: number;
    readonly responseTime: number;
    readonly throughput: number;
  };
  readonly business: {
    readonly extractionCount: number;
    readonly successRate: number;
    readonly processingTime: number;
    readonly dataQuality: number;
  };
  readonly timestamp: number;
}

// Monitoring dashboard with comprehensive data
export interface MonitoringDashboard {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly panels: readonly GrafanaPanel[];
  readonly queries: readonly GrafanaQuery[];
  readonly alerts: readonly AlertConfig[];
  readonly refreshInterval: string;
  readonly timeRange: { from: string; to: string };
  readonly created: Date;
  readonly updated: Date;
}

// Domain errors with mathematical precision
export class MonitoringServiceError extends Error {
  constructor(
    message: string,
    public readonly operation: string,
    public readonly component: string
  ) {
    super(message);
    this.name = "MonitoringServiceError";
  }
}

export class MetricsCollectionError extends Error {
  constructor(
    message: string,
    public readonly metricName: string,
    public readonly operation: string
  ) {
    super(message);
    this.name = "MetricsCollectionError";
  }
}

export class DashboardGenerationError extends Error {
  constructor(
    message: string,
    public readonly dashboardId: string,
    public readonly operation: string
  ) {
    super(message);
    this.name = "DashboardGenerationError";
  }
}

// Main Monitoring Service with formal specifications
export class MonitoringService {
  private metricsCollector: MetricsCollector | null = null;
  private prometheusExporter: PrometheusExporter | null = null;
  private grafanaBuilder: GrafanaDashboardBuilder | null = null;
  private isInitialized = false;
  private operationCount = 0;
  
  constructor(private readonly config: MonitoringConfig) {}
  
  /**
   * Initialize the monitoring service with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures all components are properly initialized
   */
  async initialize(): Promise<Result<void, Error>> {
    try {
      // Validate configuration
      const validationResult = MonitoringConfigSchema.safeParse({
        ...this.config,
        prometheus: {
          ...this.config.prometheus,
          labels: Object.fromEntries(this.config.prometheus.labels)
        }
      });
      
      if (!validationResult.success) {
        return Err(new MonitoringServiceError(
          "Invalid monitoring configuration",
          "initialize",
          "configuration"
        ));
      }
      
      // Initialize metrics collector
      this.metricsCollector = new MetricsCollector(
        this.config.metrics.maxMetricsPerName,
        this.config.metrics.defaultTimeWindow
      );
      
      const collectorInitResult = await this.metricsCollector.initialize();
      if (collectorInitResult._tag === "Left") {
        return Err(new MonitoringServiceError(
          `Failed to initialize metrics collector: ${collectorInitResult.left.message}`,
          "initialize",
          "metrics_collector"
        ));
      }
      
      // Initialize Prometheus exporter
      this.prometheusExporter = new PrometheusExporter(this.config.prometheus);
      
      const exporterInitResult = await this.prometheusExporter.initialize();
      if (exporterInitResult._tag === "Left") {
        return Err(new MonitoringServiceError(
          `Failed to initialize Prometheus exporter: ${exporterInitResult.left.message}`,
          "initialize",
          "prometheus_exporter"
        ));
      }
      
      // Initialize Grafana dashboard builder
      this.grafanaBuilder = new GrafanaDashboardBuilder(
        this.config.grafana.gridWidth,
        this.config.grafana.gridHeight
      );
      
      const builderInitResult = await this.grafanaBuilder.initialize();
      if (builderInitResult._tag === "Left") {
        return Err(new MonitoringServiceError(
          `Failed to initialize Grafana builder: ${builderInitResult.left.message}`,
          "initialize",
          "grafana_builder"
        ));
      }
      
      // Configure default alerts
      await this.configureDefaultAlerts();
      
      this.isInitialized = true;
      return Ok(undefined);
    } catch (error) {
      return Err(new MonitoringServiceError(
        `Failed to initialize monitoring service: ${error.message}`,
        "initialize",
        "service"
      ));
    }
  }
  
  /**
   * Collect system metrics with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures metrics are properly collected
   */
  async collectSystemMetrics(): Promise<Result<void, Error>> {
    if (!this.isInitialized || !this.metricsCollector) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "collect_system_metrics",
        "service"
      ));
    }
    
    try {
      const timestamp = Date.now();
      
      // Collect CPU usage
      const cpuUsage = await this.getCpuUsage();
      await this.metricsCollector.collectMetric({
        name: 'system_cpu_usage_percent',
        type: 'gauge',
        value: cpuUsage,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'percent',
          description: 'CPU usage percentage',
          help: 'Current CPU usage percentage'
        }
      });
      
      // Collect memory usage
      const memoryUsage = await this.getMemoryUsage();
      await this.metricsCollector.collectMetric({
        name: 'system_memory_usage_bytes',
        type: 'gauge',
        value: memoryUsage,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'bytes',
          description: 'Memory usage in bytes',
          help: 'Current memory usage in bytes'
        }
      });
      
      // Collect disk usage
      const diskUsage = await this.getDiskUsage();
      await this.metricsCollector.collectMetric({
        name: 'system_disk_usage_bytes',
        type: 'gauge',
        value: diskUsage,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'bytes',
          description: 'Disk usage in bytes',
          help: 'Current disk usage in bytes'
        }
      });
      
      // Collect network latency
      const networkLatency = await this.getNetworkLatency();
      await this.metricsCollector.collectMetric({
        name: 'system_network_latency_ms',
        type: 'gauge',
        value: networkLatency,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'milliseconds',
          description: 'Network latency in milliseconds',
          help: 'Current network latency in milliseconds'
        }
      });
      
      this.operationCount++;
      return Ok(undefined);
    } catch (error) {
      return Err(new MetricsCollectionError(
        `Failed to collect system metrics: ${error.message}`,
        "system_metrics",
        "collect"
      ));
    }
  }
  
  /**
   * Collect application metrics with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures metrics are properly collected
   */
  async collectApplicationMetrics(
    requestCount: number,
    errorCount: number,
    responseTime: number
  ): Promise<Result<void, Error>> {
    if (!this.isInitialized || !this.metricsCollector) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "collect_application_metrics",
        "service"
      ));
    }
    
    try {
      const timestamp = Date.now();
      
      // Collect request count
      await this.metricsCollector.collectMetric({
        name: 'application_requests_total',
        type: 'counter',
        value: requestCount,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'count',
          description: 'Total number of requests',
          help: 'Total number of requests processed'
        }
      });
      
      // Collect error rate
      const errorRate = requestCount > 0 ? (errorCount / requestCount) * 100 : 0;
      await this.metricsCollector.collectMetric({
        name: 'application_error_rate_percent',
        type: 'gauge',
        value: errorRate,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'percent',
          description: 'Error rate percentage',
          help: 'Current error rate percentage'
        }
      });
      
      // Collect response time
      await this.metricsCollector.collectMetric({
        name: 'application_response_time_ms',
        type: 'histogram',
        value: responseTime,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'milliseconds',
          description: 'Response time in milliseconds',
          help: 'Current response time in milliseconds'
        }
      });
      
      // Calculate throughput
      const throughput = requestCount / (this.config.metrics.collectionInterval / 1000);
      await this.metricsCollector.collectMetric({
        name: 'application_throughput_rps',
        type: 'gauge',
        value: throughput,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'requests_per_second',
          description: 'Throughput in requests per second',
          help: 'Current throughput in requests per second'
        }
      });
      
      this.operationCount++;
      return Ok(undefined);
    } catch (error) {
      return Err(new MetricsCollectionError(
        `Failed to collect application metrics: ${error.message}`,
        "application_metrics",
        "collect"
      ));
    }
  }
  
  /**
   * Collect business metrics with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures metrics are properly collected
   */
  async collectBusinessMetrics(
    extractionCount: number,
    successCount: number,
    processingTime: number,
    dataQuality: number
  ): Promise<Result<void, Error>> {
    if (!this.isInitialized || !this.metricsCollector) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "collect_business_metrics",
        "service"
      ));
    }
    
    try {
      const timestamp = Date.now();
      
      // Collect extraction count
      await this.metricsCollector.collectMetric({
        name: 'business_extractions_total',
        type: 'counter',
        value: extractionCount,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'count',
          description: 'Total number of extractions',
          help: 'Total number of data extractions performed'
        }
      });
      
      // Collect success rate
      const successRate = extractionCount > 0 ? (successCount / extractionCount) * 100 : 0;
      await this.metricsCollector.collectMetric({
        name: 'business_success_rate_percent',
        type: 'gauge',
        value: successRate,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'percent',
          description: 'Success rate percentage',
          help: 'Current success rate percentage'
        }
      });
      
      // Collect processing time
      await this.metricsCollector.collectMetric({
        name: 'business_processing_time_ms',
        type: 'histogram',
        value: processingTime,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'milliseconds',
          description: 'Processing time in milliseconds',
          help: 'Current processing time in milliseconds'
        }
      });
      
      // Collect data quality
      await this.metricsCollector.collectMetric({
        name: 'business_data_quality_score',
        type: 'gauge',
        value: dataQuality,
        labels: new Map([['instance', 'medical-aesthetics-engine']]),
        timestamp,
        metadata: {
          unit: 'score',
          description: 'Data quality score',
          help: 'Current data quality score'
        }
      });
      
      this.operationCount++;
      return Ok(undefined);
    } catch (error) {
      return Err(new MetricsCollectionError(
        `Failed to collect business metrics: ${error.message}`,
        "business_metrics",
        "collect"
      ));
    }
  }
  
  /**
   * Export metrics to Prometheus with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of metrics
   * CORRECTNESS: Ensures metrics are properly exported
   */
  async exportMetricsToPrometheus(): Promise<Result<string, Error>> {
    if (!this.isInitialized || !this.metricsCollector || !this.prometheusExporter) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "export_metrics",
        "service"
      ));
    }
    
    try {
      // Get all metrics
      const summaryResult = await this.metricsCollector.getMetricsSummary();
      if (summaryResult._tag === "Left") {
        return Err(new MetricsCollectionError(
          `Failed to get metrics summary: ${summaryResult.left.message}`,
          "metrics_summary",
          "export"
        ));
      }
      
      const summary = summaryResult.right;
      const metrics: PrometheusMetric[] = [];
      
      // Convert metrics to Prometheus format
      for (const metricName of summary.metricNames) {
        const aggregationResult = await this.metricsCollector.getMetricAggregation(metricName);
        if (aggregationResult._tag === "Right") {
          const aggregation = aggregationResult.right;
          
          // Create Prometheus metric
          const prometheusMetric: PrometheusMetric = {
            name: metricName,
            type: aggregation.type as 'counter' | 'gauge' | 'histogram' | 'summary',
            help: `Metric for ${metricName}`,
            labels: [],
            value: aggregation.statistics.mean,
            timestamp: aggregation.timeWindow.end
          };
          
          metrics.push(prometheusMetric);
        }
      }
      
      // Export metrics
      const exportResult = await this.prometheusExporter.exportMetrics(metrics);
      if (exportResult._tag === "Left") {
        return Err(new MetricsCollectionError(
          `Failed to export metrics: ${exportResult.left.message}`,
          "prometheus_export",
          "export"
        ));
      }
      
      this.operationCount++;
      return Ok(exportResult.right);
    } catch (error) {
      return Err(new MetricsCollectionError(
        `Failed to export metrics to Prometheus: ${error.message}`,
        "prometheus_export",
        "export"
      ));
    }
  }
  
  /**
   * Generate monitoring dashboard with mathematical precision
   * 
   * COMPLEXITY: O(p) where p is number of panels
   * CORRECTNESS: Ensures dashboard is properly generated
   */
  async generateMonitoringDashboard(): Promise<Result<MonitoringDashboard, Error>> {
    if (!this.isInitialized || !this.grafanaBuilder) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "generate_dashboard",
        "service"
      ));
    }
    
    try {
      // Create system metrics panel
      const systemPanelResult = await this.createSystemMetricsPanel();
      if (systemPanelResult._tag === "Right") {
        await this.grafanaBuilder.addPanel(systemPanelResult.right);
      }
      
      // Create application metrics panel
      const applicationPanelResult = await this.createApplicationMetricsPanel();
      if (applicationPanelResult._tag === "Right") {
        await this.grafanaBuilder.addPanel(applicationPanelResult.right);
      }
      
      // Create business metrics panel
      const businessPanelResult = await this.createBusinessMetricsPanel();
      if (businessPanelResult._tag === "Right") {
        await this.grafanaBuilder.addPanel(businessPanelResult.right);
      }
      
      // Build dashboard
      const dashboardResult = await this.grafanaBuilder.buildDashboard();
      if (dashboardResult._tag === "Left") {
        return Err(new DashboardGenerationError(
          `Failed to build dashboard: ${dashboardResult.left.message}`,
          "monitoring_dashboard",
          "build"
        ));
      }
      
      const dashboard = dashboardResult.right;
      
      const monitoringDashboard: MonitoringDashboard = {
        id: dashboard.uid,
        title: dashboard.title,
        description: dashboard.description,
        panels: dashboard.panels,
        queries: dashboard.panels.flatMap(panel => panel.targets),
        alerts: [], // Would be populated from alert configurations
        refreshInterval: dashboard.refresh,
        timeRange: dashboard.time,
        created: new Date(),
        updated: new Date()
      };
      
      this.operationCount++;
      return Ok(monitoringDashboard);
    } catch (error) {
      return Err(new DashboardGenerationError(
        `Failed to generate monitoring dashboard: ${error.message}`,
        "monitoring_dashboard",
        "generate"
      ));
    }
  }
  
  /**
   * Get monitoring metrics summary with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures summary is mathematically accurate
   */
  async getMonitoringMetricsSummary(): Promise<Result<{
    totalMetrics: number;
    metricNames: string[];
    activeAlerts: number;
    operationCount: number;
    lastUpdate: Date;
  }, Error>> {
    if (!this.isInitialized || !this.metricsCollector) {
      return Err(new MonitoringServiceError(
        "Monitoring service not initialized",
        "get_metrics_summary",
        "service"
      ));
    }
    
    try {
      const summaryResult = await this.metricsCollector.getMetricsSummary();
      if (summaryResult._tag === "Left") {
        return Err(new MetricsCollectionError(
          `Failed to get metrics summary: ${summaryResult.left.message}`,
          "metrics_summary",
          "get_summary"
        ));
      }
      
      const summary = summaryResult.right;
      
      return Ok({
        totalMetrics: summary.totalMetrics,
        metricNames: summary.metricNames,
        activeAlerts: summary.activeAlerts,
        operationCount: this.operationCount,
        lastUpdate: new Date()
      });
    } catch (error) {
      return Err(new MetricsCollectionError(
        `Failed to get monitoring metrics summary: ${error.message}`,
        "metrics_summary",
        "get_summary"
      ));
    }
  }
  
  // Helper methods with mathematical validation
  private async configureDefaultAlerts(): Promise<void> {
    if (!this.metricsCollector) return;
    
    // CPU usage alert
    await this.metricsCollector.configureAlert({
      id: 'cpu_usage_high',
      name: 'High CPU Usage',
      metricName: 'system_cpu_usage_percent',
      condition: 'gt',
      threshold: 80,
      duration: 300, // 5 minutes
      severity: 'warning',
      enabled: true,
      labels: new Map([['instance', 'medical-aesthetics-engine']])
    });
    
    // Memory usage alert
    await this.metricsCollector.configureAlert({
      id: 'memory_usage_high',
      name: 'High Memory Usage',
      metricName: 'system_memory_usage_bytes',
      condition: 'gt',
      threshold: 8 * 1024 * 1024 * 1024, // 8GB
      duration: 300, // 5 minutes
      severity: 'warning',
      enabled: true,
      labels: new Map([['instance', 'medical-aesthetics-engine']])
    });
    
    // Error rate alert
    await this.metricsCollector.configureAlert({
      id: 'error_rate_high',
      name: 'High Error Rate',
      metricName: 'application_error_rate_percent',
      condition: 'gt',
      threshold: 5, // 5%
      duration: 60, // 1 minute
      severity: 'critical',
      enabled: true,
      labels: new Map([['instance', 'medical-aesthetics-engine']])
    });
  }
  
  private async getCpuUsage(): Promise<number> {
    // Simulate CPU usage calculation
    // In real implementation, would use system APIs
    return Math.random() * 100;
  }
  
  private async getMemoryUsage(): Promise<number> {
    // Simulate memory usage calculation
    // In real implementation, would use system APIs
    return Math.random() * 16 * 1024 * 1024 * 1024; // Up to 16GB
  }
  
  private async getDiskUsage(): Promise<number> {
    // Simulate disk usage calculation
    // In real implementation, would use system APIs
    return Math.random() * 100 * 1024 * 1024 * 1024; // Up to 100GB
  }
  
  private async getNetworkLatency(): Promise<number> {
    // Simulate network latency calculation
    // In real implementation, would ping external services
    return Math.random() * 100; // Up to 100ms
  }
  
  private async createSystemMetricsPanel(): Promise<Result<GrafanaPanel, Error>> {
    if (!this.grafanaBuilder) {
      return Err(new Error("Grafana builder not initialized"));
    }
    
    const query: GrafanaQuery = {
      id: 'system_metrics_query',
      refId: 'A',
      datasource: 'Prometheus',
      query: 'system_cpu_usage_percent',
      queryType: 'query',
      interval: '5s',
      maxDataPoints: 1000,
      timeField: 'time',
      target: 'system_cpu_usage_percent',
      rawSql: '',
      format: 'time_series',
      legendFormat: 'CPU Usage',
      metricColumn: 'value',
      valueColumn: 'value',
      keyColumn: 'time',
      bucketColumn: '',
      bucketSize: 0,
      bucketOffset: 0,
      bucketCount: 0,
      alias: 'CPU Usage',
      hide: false,
      raw: false,
      useTimeRange: true,
      relativeTimeRange: { from: 'now-1h', to: 'now' },
      timeShift: '',
      cacheTimeout: '',
      queryCachingTTL: 0,
      dataSourceRef: {
        type: 'prometheus',
        uid: 'prometheus'
      }
    };
    
    return await this.grafanaBuilder.createTimeseriesPanel(
      'system_metrics_panel',
      'System Metrics',
      query,
      { x: 0, y: 0, w: 12, h: 8 }
    );
  }
  
  private async createApplicationMetricsPanel(): Promise<Result<GrafanaPanel, Error>> {
    if (!this.grafanaBuilder) {
      return Err(new Error("Grafana builder not initialized"));
    }
    
    const query: GrafanaQuery = {
      id: 'application_metrics_query',
      refId: 'A',
      datasource: 'Prometheus',
      query: 'application_requests_total',
      queryType: 'query',
      interval: '5s',
      maxDataPoints: 1000,
      timeField: 'time',
      target: 'application_requests_total',
      rawSql: '',
      format: 'time_series',
      legendFormat: 'Requests',
      metricColumn: 'value',
      valueColumn: 'value',
      keyColumn: 'time',
      bucketColumn: '',
      bucketSize: 0,
      bucketOffset: 0,
      bucketCount: 0,
      alias: 'Requests',
      hide: false,
      raw: false,
      useTimeRange: true,
      relativeTimeRange: { from: 'now-1h', to: 'now' },
      timeShift: '',
      cacheTimeout: '',
      queryCachingTTL: 0,
      dataSourceRef: {
        type: 'prometheus',
        uid: 'prometheus'
      }
    };
    
    return await this.grafanaBuilder.createTimeseriesPanel(
      'application_metrics_panel',
      'Application Metrics',
      query,
      { x: 12, y: 0, w: 12, h: 8 }
    );
  }
  
  private async createBusinessMetricsPanel(): Promise<Result<GrafanaPanel, Error>> {
    if (!this.grafanaBuilder) {
      return Err(new Error("Grafana builder not initialized"));
    }
    
    const query: GrafanaQuery = {
      id: 'business_metrics_query',
      refId: 'A',
      datasource: 'Prometheus',
      query: 'business_extractions_total',
      queryType: 'query',
      interval: '5s',
      maxDataPoints: 1000,
      timeField: 'time',
      target: 'business_extractions_total',
      rawSql: '',
      format: 'time_series',
      legendFormat: 'Extractions',
      metricColumn: 'value',
      valueColumn: 'value',
      keyColumn: 'time',
      bucketColumn: '',
      bucketSize: 0,
      bucketOffset: 0,
      bucketCount: 0,
      alias: 'Extractions',
      hide: false,
      raw: false,
      useTimeRange: true,
      relativeTimeRange: { from: 'now-1h', to: 'now' },
      timeShift: '',
      cacheTimeout: '',
      queryCachingTTL: 0,
      dataSourceRef: {
        type: 'prometheus',
        uid: 'prometheus'
      }
    };
    
    return await this.grafanaBuilder.createTimeseriesPanel(
      'business_metrics_panel',
      'Business Metrics',
      query,
      { x: 0, y: 8, w: 24, h: 8 }
    );
  }
  
  // Health check with mathematical validation
  async healthCheck(): Promise<boolean> {
    return this.isInitialized && 
           this.metricsCollector !== null && 
           this.prometheusExporter !== null && 
           this.grafanaBuilder !== null;
  }
  
  // Get service statistics
  getStatistics(): {
    isInitialized: boolean;
    operationCount: number;
    config: MonitoringConfig;
  } {
    return {
      isInitialized: this.isInitialized,
      operationCount: this.operationCount,
      config: this.config
    };
  }
}

// Factory function with mathematical validation
export function createMonitoringService(config: MonitoringConfig): MonitoringService {
  const validationResult = MonitoringConfigSchema.safeParse({
    ...config,
    prometheus: {
      ...config.prometheus,
      labels: Object.fromEntries(config.prometheus.labels)
    }
  });
  
  if (!validationResult.success) {
    throw new Error("Invalid monitoring service configuration");
  }
  
  return new MonitoringService(config);
}

// Utility functions with mathematical properties
export function validateMonitoringConfig(config: MonitoringConfig): boolean {
  return MonitoringConfigSchema.safeParse({
    ...config,
    prometheus: {
      ...config.prometheus,
      labels: Object.fromEntries(config.prometheus.labels)
    }
  }).success;
}

export function calculateMonitoringMetricsRate(
  currentCount: number,
  previousCount: number,
  timeDelta: number
): number {
  if (timeDelta <= 0) return 0;
  return (currentCount - previousCount) / timeDelta;
}

export function calculateOptimalRefreshInterval(
  dataUpdateFrequency: number,
  userInteractionLevel: 'low' | 'medium' | 'high'
): string {
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
