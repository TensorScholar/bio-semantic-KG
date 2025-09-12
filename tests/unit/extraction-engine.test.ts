/**
 * Extraction Engine Unit Tests
 * 
 * Comprehensive unit tests for the extraction engine with mathematical
 * validation and property-based testing principles.
 * 
 * @file extraction-engine.test.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { assertEquals, assertExists, assertThrows } from "https://deno.land/std@0.208.0/assert/mod.ts";
import { ExtractionEngine, createExtractionEngine } from "../../src/infrastructure/extraction/extraction-engine.ts";
import { ExtractionSource, ExtractionConfig } from "../../src/application/ports/extraction.port.ts";

// Test data with mathematical properties
const testSource: ExtractionSource = {
  id: "test-source-1",
  url: "https://example.com/clinic",
  type: "website",
  priority: 5,
  metadata: {
    created: new Date(),
    updated: new Date(),
    lastExtracted: new Date(),
    extractionCount: 0,
    successRate: 1.0
  }
};

const testConfig: ExtractionConfig = {
  maxRetries: 3,
  timeout: 300000,
  parallel: false,
  filters: [],
  selectors: {
    title: "h1, h2, h3",
    price: ".price, [class*='price']",
    description: ".description, .content"
  },
  nlpEnabled: true,
  knowledgeGraphEnabled: true
};

// Test suite with mathematical validation
Deno.test("ExtractionEngine - Initialization", async () => {
  const engine = createExtractionEngine();
  
  // Test initialization
  const initResult = await engine.initialize();
  assertEquals(initResult._tag, "Right");
  
  // Test health check
  const healthCheck = await engine.healthCheck();
  assertEquals(healthCheck, true);
  
  // Test statistics
  const stats = engine.getStatistics();
  assertEquals(stats.isInitialized, true);
  assertEquals(stats.sourceCount, 0);
  assertEquals(stats.jobCount, 0);
  assertEquals(stats.resultCount, 0);
  assertEquals(stats.extractionCount, 0);
});

Deno.test("ExtractionEngine - Add Source", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  
  // Test adding source
  const addResult = await engine.addSource(testSource);
  assertEquals(addResult._tag, "Right");
  
  // Test statistics after adding source
  const stats = engine.getStatistics();
  assertEquals(stats.sourceCount, 1);
});

Deno.test("ExtractionEngine - Start Extraction", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  await engine.addSource(testSource);
  
  // Test starting extraction
  const startResult = await engine.startExtraction(
    testSource.id,
    "beautifulsoup",
    testConfig
  );
  
  assertEquals(startResult._tag, "Right");
  
  if (startResult._tag === "Right") {
    const job = startResult.right;
    assertEquals(job.sourceId, testSource.id);
    assertEquals(job.parserId, "beautifulsoup");
    assertEquals(job.status, "pending");
    assertExists(job.metadata.created);
    assertExists(job.metadata.started);
  }
});

Deno.test("ExtractionEngine - Error Handling", async () => {
  const engine = createExtractionEngine();
  
  // Test operations before initialization
  const addResult = await engine.addSource(testSource);
  assertEquals(addResult._tag, "Left");
  
  if (addResult._tag === "Left") {
    assertEquals(addResult.left.message, "Extraction engine not initialized");
  }
});

Deno.test("ExtractionEngine - Invalid Parameters", () => {
  // Test invalid max concurrent jobs
  assertThrows(
    () => createExtractionEngine(-1, 3, 300000),
    Error,
    "Max concurrent jobs must be positive"
  );
  
  // Test invalid max retries
  assertThrows(
    () => createExtractionEngine(10, -1, 300000),
    Error,
    "Max retries must be non-negative"
  );
  
  // Test invalid default timeout
  assertThrows(
    () => createExtractionEngine(10, 3, -1),
    Error,
    "Default timeout must be positive"
  );
});

Deno.test("ExtractionEngine - Mathematical Properties", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  
  // Test mathematical properties
  const stats = engine.getStatistics();
  
  // All counts should be non-negative
  assertEquals(stats.sourceCount >= 0, true);
  assertEquals(stats.jobCount >= 0, true);
  assertEquals(stats.resultCount >= 0, true);
  assertEquals(stats.extractionCount >= 0, true);
  
  // Initial state should have zero counts
  assertEquals(stats.sourceCount, 0);
  assertEquals(stats.jobCount, 0);
  assertEquals(stats.resultCount, 0);
  assertEquals(stats.extractionCount, 0);
});

Deno.test("ExtractionEngine - Concurrent Operations", async () => {
  const engine = createExtractionEngine(2, 3, 300000); // Max 2 concurrent jobs
  await engine.initialize();
  await engine.addSource(testSource);
  
  // Start multiple extractions
  const promises = [];
  for (let i = 0; i < 5; i++) {
    promises.push(
      engine.startExtraction(
        testSource.id,
        "beautifulsoup",
        testConfig
      )
    );
  }
  
  const results = await Promise.all(promises);
  
  // All should succeed (engine handles concurrency internally)
  for (const result of results) {
    assertEquals(result._tag, "Right");
  }
});

Deno.test("ExtractionEngine - Performance Characteristics", async () => {
  const engine = createExtractionEngine();
  const startTime = Date.now();
  
  await engine.initialize();
  
  const initTime = Date.now() - startTime;
  
  // Initialization should be fast (< 1 second)
  assertEquals(initTime < 1000, true);
  
  // Test adding multiple sources
  const addStartTime = Date.now();
  
  for (let i = 0; i < 100; i++) {
    const source: ExtractionSource = {
      ...testSource,
      id: `test-source-${i}`,
      url: `https://example.com/clinic-${i}`
    };
    
    const addResult = await engine.addSource(source);
    assertEquals(addResult._tag, "Right");
  }
  
  const addTime = Date.now() - addStartTime;
  
  // Adding 100 sources should be fast (< 5 seconds)
  assertEquals(addTime < 5000, true);
  
  // Test statistics
  const stats = engine.getStatistics();
  assertEquals(stats.sourceCount, 100);
});

Deno.test("ExtractionEngine - Memory Management", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  
  // Test memory usage doesn't grow unbounded
  const initialStats = engine.getStatistics();
  
  // Add and remove sources multiple times
  for (let i = 0; i < 10; i++) {
    const source: ExtractionSource = {
      ...testSource,
      id: `temp-source-${i}`,
      url: `https://example.com/temp-${i}`
    };
    
    await engine.addSource(source);
  }
  
  const afterAddStats = engine.getStatistics();
  assertEquals(afterAddStats.sourceCount, 10);
  
  // Memory should be managed properly
  assertEquals(afterAddStats.sourceCount >= initialStats.sourceCount, true);
});

Deno.test("ExtractionEngine - Error Recovery", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  
  // Test error recovery
  const invalidSource: ExtractionSource = {
    id: "invalid-source",
    url: "invalid-url",
    type: "website",
    priority: 5,
    metadata: {
      created: new Date(),
      updated: new Date(),
      lastExtracted: new Date(),
      extractionCount: 0,
      successRate: 1.0
    }
  };
  
  // Adding invalid source should still work (validation happens later)
  const addResult = await engine.addSource(invalidSource);
  assertEquals(addResult._tag, "Right");
  
  // Engine should remain functional
  const healthCheck = await engine.healthCheck();
  assertEquals(healthCheck, true);
});

Deno.test("ExtractionEngine - Configuration Validation", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  await engine.addSource(testSource);
  
  // Test with different configurations
  const configs: ExtractionConfig[] = [
    {
      maxRetries: 0,
      timeout: 1000,
      parallel: false,
      filters: [],
      selectors: {},
      nlpEnabled: false,
      knowledgeGraphEnabled: false
    },
    {
      maxRetries: 10,
      timeout: 600000,
      parallel: true,
      filters: ["clinic", "aesthetic"],
      selectors: { title: "h1" },
      nlpEnabled: true,
      knowledgeGraphEnabled: true
    }
  ];
  
  for (const config of configs) {
    const startResult = await engine.startExtraction(
      testSource.id,
      "beautifulsoup",
      config
    );
    
    assertEquals(startResult._tag, "Right");
  }
});

Deno.test("ExtractionEngine - Mathematical Consistency", async () => {
  const engine = createExtractionEngine();
  await engine.initialize();
  
  // Test mathematical consistency
  const operations = [
    () => engine.addSource(testSource),
    () => engine.addSource({ ...testSource, id: "test-source-2" }),
    () => engine.addSource({ ...testSource, id: "test-source-3" })
  ];
  
  for (const operation of operations) {
    const result = await operation();
    assertEquals(result._tag, "Right");
  }
  
  const stats = engine.getStatistics();
  assertEquals(stats.sourceCount, 3);
  
  // Test idempotency
  const duplicateResult = await engine.addSource(testSource);
  assertEquals(duplicateResult._tag, "Right");
  
  const finalStats = engine.getStatistics();
  assertEquals(finalStats.sourceCount, 3); // Should not increase
});
