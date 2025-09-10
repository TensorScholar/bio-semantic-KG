/**
 * Medical Aesthetics Extraction Engine - Main Entry Point
 * 
 * Elite Technical Consortium Implementation
 * Advanced medical data extraction system with bilingual NLP capabilities,
 * knowledge graph construction, and comprehensive analytics.
 * 
 * @file index.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { Application, Router, Context } from "@oak/mod.ts";
import { config } from "@std/dotenv/mod.ts";
import { logger } from "@std/log/mod.ts";
import { serve } from "@std/http/server.ts";

// Core domain imports
import { MedicalClinicAggregate } from "./core/entities/medical-clinic.ts";
import { MedicalProcedureAggregate } from "./core/entities/medical-procedure.ts";
import { Price } from "./core/value-objects/price.ts";
import { Rating } from "./core/value-objects/rating.ts";
import { URL } from "./core/value-objects/url.ts";

// Application services
import { ExtractionService } from "./application/services/extraction.service.ts";
import { PersistenceService } from "./application/services/persistence.service.ts";
import { TransformationService } from "./application/services/transformation.service.ts";
import { CQRSService } from "./application/services/cqrs.service.ts";

// Infrastructure
import { Neo4jRepository } from "./infrastructure/persistence/neo4j/neo4j.repository.ts";
import { ElasticsearchRepository } from "./infrastructure/persistence/elasticsearch/elasticsearch.repository.ts";
import { RedisCache } from "./infrastructure/caching/redis.cache.ts";
import { TelemetrySystem } from "./infrastructure/monitoring/telemetry.ts";
import { SecurityManager } from "./infrastructure/security/security.manager.ts";

// Shared utilities
import { Result, Ok, Err } from "./shared/kernel/result.ts";
import { Option, Some, None } from "./shared/kernel/option.ts";
import { Either, Left, Right } from "./shared/kernel/either.ts";

// Configuration interface
interface AppConfig {
  readonly port: number;
  readonly metricsPort: number;
  readonly logLevel: string;
  readonly environment: string;
  readonly neo4j: {
    readonly uri: string;
    readonly username: string;
    readonly password: string;
  };
  readonly elasticsearch: {
    readonly url: string;
    readonly username?: string;
    readonly password?: string;
  };
  readonly redis: {
    readonly url: string;
    readonly password?: string;
  };
  readonly timescale: {
    readonly url: string;
  };
  readonly browserless: {
    readonly url: string;
    readonly token: string;
  };
  readonly security: {
    readonly jwtSecret: string;
    readonly encryptionKey: string;
  };
  readonly monitoring: {
    readonly prometheusEnabled: boolean;
    readonly jaegerEnabled: boolean;
    readonly grafanaEnabled: boolean;
  };
}

// Application state
class ApplicationState {
  private static instance: ApplicationState;
  private config: AppConfig | null = null;
  private services: Map<string, any> = new Map();
  private isInitialized = false;

  private constructor() {}

  static getInstance(): ApplicationState {
    if (!ApplicationState.instance) {
      ApplicationState.instance = new ApplicationState();
    }
    return ApplicationState.instance;
  }

  async initialize(): Promise<Result<void, Error>> {
    try {
      // Load environment configuration
      await config({ export: true });
      
      this.config = {
        port: parseInt(Deno.env.get("PORT") || "8080"),
        metricsPort: parseInt(Deno.env.get("METRICS_PORT") || "9090"),
        logLevel: Deno.env.get("LOG_LEVEL") || "info",
        environment: Deno.env.get("NODE_ENV") || "development",
        neo4j: {
          uri: Deno.env.get("NEO4J_URI") || "bolt://localhost:7687",
          username: Deno.env.get("NEO4J_USER") || "neo4j",
          password: Deno.env.get("NEO4J_PASSWORD") || "password"
        },
        elasticsearch: {
          url: Deno.env.get("ELASTICSEARCH_URL") || "http://localhost:9200",
          username: Deno.env.get("ELASTICSEARCH_USERNAME"),
          password: Deno.env.get("ELASTICSEARCH_PASSWORD")
        },
        redis: {
          url: Deno.env.get("REDIS_URL") || "redis://localhost:6379",
          password: Deno.env.get("REDIS_PASSWORD")
        },
        timescale: {
          url: Deno.env.get("TIMESCALE_URL") || "postgresql://postgres:password@localhost:5432/medical"
        },
        browserless: {
          url: Deno.env.get("BROWSERLESS_URL") || "http://localhost:3000",
          token: Deno.env.get("BROWSERLESS_TOKEN") || "token"
        },
        security: {
          jwtSecret: Deno.env.get("JWT_SECRET") || "default-secret",
          encryptionKey: Deno.env.get("ENCRYPTION_KEY") || "default-encryption-key"
        },
        monitoring: {
          prometheusEnabled: Deno.env.get("PROMETHEUS_ENABLED") === "true",
          jaegerEnabled: Deno.env.get("JAEGER_ENABLED") === "true",
          grafanaEnabled: Deno.env.get("GRAFANA_ENABLED") === "true"
        }
      };

      // Initialize logging
      await logger.setup({
        handlers: {
          console: new logger.ConsoleHandler(this.config.logLevel, {
            formatter: this.config.environment === "production" 
              ? "{datetime} {levelName} {msg}" 
              : "{datetime} {levelName} {loggerName} {msg}"
          })
        },
        loggers: {
          default: {
            level: this.config.logLevel,
            handlers: ["console"]
          }
        }
      });

      // Initialize services
      await this.initializeServices();

      this.isInitialized = true;
      logger.info("Application initialized successfully");
      
      return Ok(undefined);
    } catch (error) {
      logger.error(`Failed to initialize application: ${error.message}`);
      return Err(error);
    }
  }

  private async initializeServices(): Promise<void> {
    if (!this.config) throw new Error("Configuration not loaded");

    // Initialize telemetry system
    const telemetry = new TelemetrySystem();
    await telemetry.initialize();
    this.services.set("telemetry", telemetry);

    // Initialize security manager
    const security = new SecurityManager(this.config.security);
    await security.initialize();
    this.services.set("security", security);

    // Initialize repositories
    const neo4jRepo = new Neo4jRepository(this.config.neo4j);
    await neo4jRepo.initialize();
    this.services.set("neo4j", neo4jRepo);

    const elasticsearchRepo = new ElasticsearchRepository(this.config.elasticsearch);
    await elasticsearchRepo.initialize();
    this.services.set("elasticsearch", elasticsearchRepo);

    const redisCache = new RedisCache(this.config.redis);
    await redisCache.initialize();
    this.services.set("redis", redisCache);

    // Initialize application services
    const extractionService = new ExtractionService(
      this.services.get("neo4j"),
      this.services.get("elasticsearch"),
      this.services.get("redis"),
      this.config.browserless
    );
    this.services.set("extraction", extractionService);

    const persistenceService = new PersistenceService(
      this.services.get("neo4j"),
      this.services.get("elasticsearch"),
      this.services.get("redis")
    );
    this.services.set("persistence", persistenceService);

    const transformationService = new TransformationService(
      this.services.get("telemetry")
    );
    this.services.set("transformation", transformationService);

    const cqrsService = new CQRSService(
      this.services.get("extraction"),
      this.services.get("persistence"),
      this.services.get("transformation")
    );
    this.services.set("cqrs", cqrsService);
  }

  getConfig(): AppConfig {
    if (!this.config) throw new Error("Application not initialized");
    return this.config;
  }

  getService<T>(name: string): T {
    if (!this.isInitialized) throw new Error("Application not initialized");
    const service = this.services.get(name);
    if (!service) throw new Error(`Service '${name}' not found`);
    return service;
  }

  isReady(): boolean {
    return this.isInitialized;
  }
}

// Health check handler
async function healthCheck(ctx: Context): Promise<void> {
  const state = ApplicationState.getInstance();
  
  if (!state.isReady()) {
    ctx.response.status = 503;
    ctx.response.body = {
      status: "unhealthy",
      message: "Application not initialized",
      timestamp: new Date().toISOString()
    };
    return;
  }

  try {
    // Check service health
    const services = {
      neo4j: await state.getService("neo4j").healthCheck(),
      elasticsearch: await state.getService("elasticsearch").healthCheck(),
      redis: await state.getService("redis").healthCheck(),
      extraction: await state.getService("extraction").healthCheck()
    };

    const allHealthy = Object.values(services).every(healthy => healthy);

    ctx.response.status = allHealthy ? 200 : 503;
    ctx.response.body = {
      status: allHealthy ? "healthy" : "unhealthy",
      services,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: "1.0.0"
    };
  } catch (error) {
    ctx.response.status = 503;
    ctx.response.body = {
      status: "unhealthy",
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
}

// Metrics handler
async function metrics(ctx: Context): Promise<void> {
  const state = ApplicationState.getInstance();
  const telemetry = state.getService("telemetry");
  
  try {
    const metrics = await telemetry.getMetrics();
    ctx.response.headers.set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
    ctx.response.body = metrics;
  } catch (error) {
    ctx.response.status = 500;
    ctx.response.body = { error: error.message };
  }
}

// Extraction endpoint
async function extractClinic(ctx: Context): Promise<void> {
  const state = ApplicationState.getInstance();
  const extractionService = state.getService("extraction");
  const telemetry = state.getService("telemetry");
  
  try {
    const { url } = await ctx.request.body().value;
    
    if (!url) {
      ctx.response.status = 400;
      ctx.response.body = { error: "URL is required" };
      return;
    }

    // Validate URL
    const urlResult = URL.create(url);
    if (urlResult._tag === "Left") {
      ctx.response.status = 400;
      ctx.response.body = { error: "Invalid URL format" };
      return;
    }

    // Check URL security
    const securityCheck = urlResult.right.isValidForExtraction();
    if (securityCheck._tag === "Left") {
      ctx.response.status = 400;
      ctx.response.body = { error: securityCheck.left.message };
      return;
    }

    // Start extraction with tracing
    const result = await telemetry.traceExtraction(
      "extract_clinic",
      () => extractionService.extractClinic(urlResult.right)
    );

    if (result._tag === "Right") {
      ctx.response.status = 200;
      ctx.response.body = {
        success: true,
        data: result.right.toJSON(),
        timestamp: new Date().toISOString()
      };
    } else {
      ctx.response.status = 500;
      ctx.response.body = {
        success: false,
        error: result.left.message,
        timestamp: new Date().toISOString()
      };
    }
  } catch (error) {
    logger.error(`Extraction error: ${error.message}`);
    ctx.response.status = 500;
    ctx.response.body = {
      success: false,
      error: "Internal server error",
      timestamp: new Date().toISOString()
    };
  }
}

// Search endpoint
async function searchClinics(ctx: Context): Promise<void> {
  const state = ApplicationState.getInstance();
  const persistenceService = state.getService("persistence");
  
  try {
    const url = new URL(ctx.request.url);
    const query = url.searchParams.get("q") || "";
    const limit = parseInt(url.searchParams.get("limit") || "10");
    const offset = parseInt(url.searchParams.get("offset") || "0");

    const result = await persistenceService.searchClinics(query, limit, offset);

    if (result._tag === "Right") {
      ctx.response.status = 200;
      ctx.response.body = {
        success: true,
        data: result.right,
        pagination: {
          limit,
          offset,
          total: result.right.length
        },
        timestamp: new Date().toISOString()
      };
    } else {
      ctx.response.status = 500;
      ctx.response.body = {
        success: false,
        error: result.left.message,
        timestamp: new Date().toISOString()
      };
    }
  } catch (error) {
    logger.error(`Search error: ${error.message}`);
    ctx.response.status = 500;
    ctx.response.body = {
      success: false,
      error: "Internal server error",
      timestamp: new Date().toISOString()
    };
  }
}

// Analytics endpoint
async function getAnalytics(ctx: Context): Promise<void> {
  const state = ApplicationState.getInstance();
  const persistenceService = state.getService("persistence");
  
  try {
    const url = new URL(ctx.request.url);
    const timeRange = url.searchParams.get("timeRange") || "7d";
    const metric = url.searchParams.get("metric") || "all";

    const result = await persistenceService.getAnalytics(timeRange, metric);

    if (result._tag === "Right") {
      ctx.response.status = 200;
      ctx.response.body = {
        success: true,
        data: result.right,
        timestamp: new Date().toISOString()
      };
    } else {
      ctx.response.status = 500;
      ctx.response.body = {
        success: false,
        error: result.left.message,
        timestamp: new Date().toISOString()
      };
    }
  } catch (error) {
    logger.error(`Analytics error: ${error.message}`);
    ctx.response.status = 500;
    ctx.response.body = {
      success: false,
      error: "Internal server error",
      timestamp: new Date().toISOString()
    };
  }
}

// Main application setup
async function createApplication(): Promise<Application> {
  const app = new Application();
  
  // CORS middleware
  app.use(async (ctx, next) => {
    ctx.response.headers.set("Access-Control-Allow-Origin", "*");
    ctx.response.headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    ctx.response.headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
    
    if (ctx.request.method === "OPTIONS") {
      ctx.response.status = 200;
      return;
    }
    
    await next();
  });

  // Request logging middleware
  app.use(async (ctx, next) => {
    const start = Date.now();
    await next();
    const duration = Date.now() - start;
    
    logger.info(`${ctx.request.method} ${ctx.request.url.pathname} - ${ctx.response.status} - ${duration}ms`);
  });

  // Security middleware
  app.use(async (ctx, next) => {
    const state = ApplicationState.getInstance();
    const security = state.getService("security");
    
    // Add security headers
    ctx.response.headers.set("X-Content-Type-Options", "nosniff");
    ctx.response.headers.set("X-Frame-Options", "DENY");
    ctx.response.headers.set("X-XSS-Protection", "1; mode=block");
    ctx.response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
    
    await next();
  });

  // API routes
  const router = new Router();
  
  // Health and monitoring
  router.get("/health", healthCheck);
  router.get("/health/live", healthCheck);
  router.get("/health/ready", healthCheck);
  router.get("/metrics", metrics);
  
  // API endpoints
  router.post("/api/v1/extract", extractClinic);
  router.get("/api/v1/search", searchClinics);
  router.get("/api/v1/analytics", getAnalytics);
  
  // API documentation
  router.get("/api/v1/docs", (ctx) => {
    ctx.response.body = {
      name: "Medical Aesthetics Extraction Engine",
      version: "1.0.0",
      description: "Advanced medical data extraction system with bilingual NLP capabilities",
      endpoints: {
        "POST /api/v1/extract": "Extract clinic data from URL",
        "GET /api/v1/search": "Search clinics with query parameters",
        "GET /api/v1/analytics": "Get analytics data",
        "GET /health": "Health check endpoint",
        "GET /metrics": "Prometheus metrics"
      },
      timestamp: new Date().toISOString()
    };
  });

  app.use(router.routes());
  app.use(router.allowedMethods());

  // 404 handler
  app.use((ctx) => {
    ctx.response.status = 404;
    ctx.response.body = {
      error: "Not Found",
      message: `Route ${ctx.request.method} ${ctx.request.url.pathname} not found`,
      timestamp: new Date().toISOString()
    };
  });

  // Error handler
  app.addEventListener("error", (event) => {
    logger.error(`Unhandled error: ${event.error.message}`);
  });

  return app;
}

// Graceful shutdown handler
async function gracefulShutdown(signal: string): Promise<void> {
  logger.info(`Received ${signal}, shutting down gracefully...`);
  
  try {
    const state = ApplicationState.getInstance();
    
    // Close all services
    const services = ["neo4j", "elasticsearch", "redis", "telemetry"];
    for (const serviceName of services) {
      try {
        const service = state.getService(serviceName);
        if (service && typeof service.close === "function") {
          await service.close();
        }
      } catch (error) {
        logger.warn(`Error closing service ${serviceName}: ${error.message}`);
      }
    }
    
    logger.info("Graceful shutdown completed");
    Deno.exit(0);
  } catch (error) {
    logger.error(`Error during shutdown: ${error.message}`);
    Deno.exit(1);
  }
}

// Main application entry point
async function main(): Promise<void> {
  try {
    logger.info("Starting Medical Aesthetics Extraction Engine...");
    
    // Initialize application state
    const state = ApplicationState.getInstance();
    const initResult = await state.initialize();
    
    if (initResult._tag === "Left") {
      logger.error(`Failed to initialize application: ${initResult.left.message}`);
      Deno.exit(1);
    }
    
    // Create application
    const app = await createApplication();
    const config = state.getConfig();
    
    // Set up graceful shutdown
    Deno.addSignalListener("SIGINT", () => gracefulShutdown("SIGINT"));
    Deno.addSignalListener("SIGTERM", () => gracefulShutdown("SIGTERM"));
    
    // Start server
    logger.info(`Server starting on port ${config.port}`);
    await serve(app.fetch, {
      port: config.port,
      hostname: "0.0.0.0"
    });
    
  } catch (error) {
    logger.error(`Failed to start application: ${error.message}`);
    Deno.exit(1);
  }
}

// Start the application
if (import.meta.main) {
  await main();
}

export { ApplicationState, main };
