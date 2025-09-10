/**
 * Medical Procedure Entity - Core Domain Model
 * 
 * Represents a medical aesthetic procedure with comprehensive classification,
 * pricing, and outcome data. Implements DDD aggregate pattern with medical
 * terminology validation.
 * 
 * @file medical-procedure.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Branded } from "../../shared/types/branded.ts";
import { Price, PriceRange } from "../value-objects/price.ts";
import { Rating } from "../value-objects/rating.ts";

// Branded types for type safety
export type ProcedureId = Branded<string, "ProcedureId">;
export type ProcedureName = Branded<string, "ProcedureName">;
export type ICD10Code = Branded<string, "ICD10Code">;
export type CPTCode = Branded<string, "CPTCode">;
export type ProcedureCategory = Branded<string, "ProcedureCategory">;

// Medical procedure categories
export const PROCEDURE_CATEGORIES = {
  SURGICAL: "surgical",
  NON_SURGICAL: "non-surgical",
  INJECTION: "injection",
  LASER: "laser",
  DIAGNOSTIC: "diagnostic",
  REHABILITATIVE: "rehabilitative",
  PREVENTIVE: "preventive"
} as const;

export type ProcedureCategoryType = typeof PROCEDURE_CATEGORIES[keyof typeof PROCEDURE_CATEGORIES];

// Procedure complexity levels
export const COMPLEXITY_LEVELS = {
  MINIMAL: "minimal",
  LOW: "low",
  MODERATE: "moderate",
  HIGH: "high",
  EXTREME: "extreme"
} as const;

export type ComplexityLevel = typeof COMPLEXITY_LEVELS[keyof typeof COMPLEXITY_LEVELS];

// Validation schemas
const ProcedureIdSchema = z.string().uuid().brand<"ProcedureId">();
const ProcedureNameSchema = z.string().min(2).max(200).brand<"ProcedureName">();
const ICD10CodeSchema = z.string().regex(/^[A-Z]\d{2}(\.\d{1,3})?$/).brand<"ICD10Code">();
const CPTCodeSchema = z.string().regex(/^\d{5}$/).brand<"CPTCode">();
const ProcedureCategorySchema = z.enum([
  "surgical", "non-surgical", "injection", "laser", 
  "diagnostic", "rehabilitative", "preventive"
]).brand<"ProcedureCategory">();

// Procedure details
export interface ProcedureDetails {
  readonly description: string;
  readonly indications: readonly string[];
  readonly contraindications: readonly string[];
  readonly risks: readonly string[];
  readonly benefits: readonly string[];
  readonly recoveryTime: string;
  readonly anesthesiaRequired: boolean;
  readonly outpatientProcedure: boolean;
}

const ProcedureDetailsSchema = z.object({
  description: z.string().min(10),
  indications: z.array(z.string().min(1)),
  contraindications: z.array(z.string().min(1)),
  risks: z.array(z.string().min(1)),
  benefits: z.array(z.string().min(1)),
  recoveryTime: z.string().min(1),
  anesthesiaRequired: z.boolean(),
  outpatientProcedure: z.boolean()
});

// Procedure pricing
export interface ProcedurePricing {
  readonly priceRange: PriceRange;
  readonly insuranceCoverage: boolean;
  readonly paymentPlans: readonly string[];
  readonly financingOptions: readonly string[];
  readonly consultationFee?: Price;
  readonly followUpFee?: Price;
}

const ProcedurePricingSchema = z.object({
  priceRange: z.any(), // Will be validated by PriceRange schema
  insuranceCoverage: z.boolean(),
  paymentPlans: z.array(z.string()),
  financingOptions: z.array(z.string()),
  consultationFee: z.any().optional(),
  followUpFee: z.any().optional()
});

// Procedure outcomes
export interface ProcedureOutcome {
  readonly successRate: number; // Percentage
  readonly satisfactionRating: Rating;
  readonly complicationRate: number; // Percentage
  readonly revisionRate: number; // Percentage
  readonly averageRecoveryTime: number; // Days
  readonly longTermResults: string;
}

const ProcedureOutcomeSchema = z.object({
  successRate: z.number().min(0).max(100),
  satisfactionRating: z.any(), // Will be validated by Rating schema
  complicationRate: z.number().min(0).max(100),
  revisionRate: z.number().min(0).max(100),
  averageRecoveryTime: z.number().positive(),
  longTermResults: z.string().min(1)
});

// Main MedicalProcedure entity
export interface MedicalProcedure {
  readonly id: ProcedureId;
  readonly name: ProcedureName;
  readonly category: ProcedureCategory;
  readonly complexity: ComplexityLevel;
  readonly icd10Codes: readonly ICD10Code[];
  readonly cptCodes: readonly CPTCode[];
  readonly details: ProcedureDetails;
  readonly pricing: ProcedurePricing;
  readonly outcome: ProcedureOutcome;
  readonly alternativeNames: readonly string[];
  readonly relatedProcedures: readonly ProcedureId[];
  readonly bodyAreas: readonly string[];
  readonly ageRestrictions?: {
    readonly minAge?: number;
    readonly maxAge?: number;
  };
  readonly genderRestrictions?: readonly ("male" | "female" | "non-binary")[];
  readonly lastUpdated: Date;
  readonly extractionMetadata: {
    readonly sourceUrl: string;
    readonly extractedAt: Date;
    readonly confidenceScore: number;
  };
}

const MedicalProcedureSchema = z.object({
  id: ProcedureIdSchema,
  name: ProcedureNameSchema,
  category: ProcedureCategorySchema,
  complexity: z.enum(["minimal", "low", "moderate", "high", "extreme"]),
  icd10Codes: z.array(ICD10CodeSchema),
  cptCodes: z.array(CPTCodeSchema),
  details: ProcedureDetailsSchema,
  pricing: ProcedurePricingSchema,
  outcome: ProcedureOutcomeSchema,
  alternativeNames: z.array(z.string()),
  relatedProcedures: z.array(ProcedureIdSchema),
  bodyAreas: z.array(z.string()),
  ageRestrictions: z.object({
    minAge: z.number().positive().optional(),
    maxAge: z.number().positive().optional()
  }).optional(),
  genderRestrictions: z.array(z.enum(["male", "female", "non-binary"])).optional(),
  lastUpdated: z.date(),
  extractionMetadata: z.object({
    sourceUrl: z.string().url(),
    extractedAt: z.date(),
    confidenceScore: z.number().min(0).max(1)
  })
});

// Domain errors
export class ProcedureValidationError extends Error {
  constructor(message: string, public readonly field: string) {
    super(message);
    this.name = "ProcedureValidationError";
  }
}

export class InvalidMedicalCodeError extends Error {
  constructor(code: string, type: "ICD10" | "CPT") {
    super(`Invalid ${type} code: ${code}`);
    this.name = "InvalidMedicalCodeError";
  }
}

export class DuplicateProcedureError extends Error {
  constructor(procedureName: string) {
    super(`Procedure '${procedureName}' already exists`);
    this.name = "DuplicateProcedureError";
  }
}

// Factory functions
export const createProcedureId = (id: string): Either<ProcedureValidationError, ProcedureId> => {
  try {
    const result = ProcedureIdSchema.parse(id);
    return right(result);
  } catch (error) {
    return left(new ProcedureValidationError("Invalid procedure ID format", "id"));
  }
};

export const createProcedureName = (name: string): Either<ProcedureValidationError, ProcedureName> => {
  try {
    const result = ProcedureNameSchema.parse(name);
    return right(result);
  } catch (error) {
    return left(new ProcedureValidationError("Invalid procedure name format", "name"));
  }
};

export const createICD10Code = (code: string): Either<InvalidMedicalCodeError, ICD10Code> => {
  try {
    const result = ICD10CodeSchema.parse(code);
    return right(result);
  } catch (error) {
    return left(new InvalidMedicalCodeError(code, "ICD10"));
  }
};

export const createCPTCode = (code: string): Either<InvalidMedicalCodeError, CPTCode> => {
  try {
    const result = CPTCodeSchema.parse(code);
    return right(result);
  } catch (error) {
    return left(new InvalidMedicalCodeError(code, "CPT"));
  }
};

export const createProcedureCategory = (category: string): Either<ProcedureValidationError, ProcedureCategory> => {
  try {
    const result = ProcedureCategorySchema.parse(category);
    return right(result);
  } catch (error) {
    return left(new ProcedureValidationError("Invalid procedure category", "category"));
  }
};

// Main MedicalProcedure class with business logic
export class MedicalProcedureAggregate {
  private constructor(private readonly procedure: MedicalProcedure) {}

  static create(procedure: MedicalProcedure): Either<ProcedureValidationError, MedicalProcedureAggregate> {
    try {
      MedicalProcedureSchema.parse(procedure);
      return right(new MedicalProcedureAggregate(procedure));
    } catch (error) {
      if (error instanceof z.ZodError) {
        const firstError = error.errors[0];
        return left(new ProcedureValidationError(
          firstError.message,
          firstError.path.join(".")
        ));
      }
      return left(new ProcedureValidationError("Unknown validation error", "procedure"));
    }
  }

  // Business invariants
  validatePricing(): Either<ProcedureValidationError, void> {
    if (this.procedure.pricing.priceRange.min > this.procedure.pricing.priceRange.max) {
      return left(new ProcedureValidationError(
        "Minimum price cannot be greater than maximum price",
        "pricing.priceRange"
      ));
    }
    return right(undefined);
  }

  validateOutcome(): Either<ProcedureValidationError, void> {
    if (this.procedure.outcome.successRate + this.procedure.outcome.complicationRate > 100) {
      return left(new ProcedureValidationError(
        "Success rate and complication rate cannot exceed 100%",
        "outcome"
      ));
    }
    return right(undefined);
  }

  validateAgeRestrictions(): Either<ProcedureValidationError, void> {
    if (this.procedure.ageRestrictions) {
      const { minAge, maxAge } = this.procedure.ageRestrictions;
      if (minAge && maxAge && minAge > maxAge) {
        return left(new ProcedureValidationError(
          "Minimum age cannot be greater than maximum age",
          "ageRestrictions"
        ));
      }
    }
    return right(undefined);
  }

  // Procedure classification
  isSurgical(): boolean {
    return this.procedure.category === "surgical";
  }

  isNonSurgical(): boolean {
    return this.procedure.category === "non-surgical";
  }

  isInjection(): boolean {
    return this.procedure.category === "injection";
  }

  isLaser(): boolean {
    return this.procedure.category === "laser";
  }

  requiresAnesthesia(): boolean {
    return this.procedure.details.anesthesiaRequired;
  }

  isOutpatient(): boolean {
    return this.procedure.details.outpatientProcedure;
  }

  // Complexity assessment
  getComplexityScore(): number {
    const complexityScores = {
      minimal: 1,
      low: 2,
      moderate: 3,
      high: 4,
      extreme: 5
    };
    return complexityScores[this.procedure.complexity];
  }

  isHighRisk(): boolean {
    return this.procedure.complexity === "high" || this.procedure.complexity === "extreme";
  }

  // Pricing analysis
  getAveragePrice(): Price {
    return this.procedure.pricing.priceRange.average;
  }

  isExpensive(): boolean {
    const averagePrice = this.getAveragePrice();
    // Consider expensive if average price > $5000
    return averagePrice.value > 5000;
  }

  isAffordable(): boolean {
    const averagePrice = this.getAveragePrice();
    // Consider affordable if average price < $1000
    return averagePrice.value < 1000;
  }

  // Outcome analysis
  isHighSuccess(): boolean {
    return this.procedure.outcome.successRate >= 90;
  }

  isLowRisk(): boolean {
    return this.procedure.outcome.complicationRate <= 5;
  }

  hasGoodSatisfaction(): boolean {
    return this.procedure.outcome.satisfactionRating.value >= 80;
  }

  // Medical code utilities
  getPrimaryICD10Code(): ICD10Code | null {
    return this.procedure.icd10Codes.length > 0 ? this.procedure.icd10Codes[0] : null;
  }

  getPrimaryCPTCode(): CPTCode | null {
    return this.procedure.cptCodes.length > 0 ? this.procedure.cptCodes[0] : null;
  }

  hasMedicalCodes(): boolean {
    return this.procedure.icd10Codes.length > 0 || this.procedure.cptCodes.length > 0;
  }

  // Related procedures
  addRelatedProcedure(procedureId: ProcedureId): MedicalProcedureAggregate {
    if (this.procedure.relatedProcedures.includes(procedureId)) {
      return this; // Already related
    }

    const updatedProcedure: MedicalProcedure = {
      ...this.procedure,
      relatedProcedures: [...this.procedure.relatedProcedures, procedureId],
      lastUpdated: new Date()
    };

    return new MedicalProcedureAggregate(updatedProcedure);
  }

  removeRelatedProcedure(procedureId: ProcedureId): MedicalProcedureAggregate {
    const updatedProcedure: MedicalProcedure = {
      ...this.procedure,
      relatedProcedures: this.procedure.relatedProcedures.filter(id => id !== procedureId),
      lastUpdated: new Date()
    };

    return new MedicalProcedureAggregate(updatedProcedure);
  }

  // Alternative names
  addAlternativeName(name: string): MedicalProcedureAggregate {
    if (this.procedure.alternativeNames.includes(name)) {
      return this; // Already exists
    }

    const updatedProcedure: MedicalProcedure = {
      ...this.procedure,
      alternativeNames: [...this.procedure.alternativeNames, name],
      lastUpdated: new Date()
    };

    return new MedicalProcedureAggregate(updatedProcedure);
  }

  // Body areas
  addBodyArea(area: string): MedicalProcedureAggregate {
    if (this.procedure.bodyAreas.includes(area)) {
      return this; // Already exists
    }

    const updatedProcedure: MedicalProcedure = {
      ...this.procedure,
      bodyAreas: [...this.procedure.bodyAreas, area],
      lastUpdated: new Date()
    };

    return new MedicalProcedureAggregate(updatedProcedure);
  }

  // Search and matching
  matchesSearchTerm(term: string): boolean {
    const searchTerm = term.toLowerCase();
    
    return (
      this.procedure.name.toLowerCase().includes(searchTerm) ||
      this.procedure.alternativeNames.some(name => 
        name.toLowerCase().includes(searchTerm)
      ) ||
      this.procedure.details.description.toLowerCase().includes(searchTerm) ||
      this.procedure.bodyAreas.some(area => 
        area.toLowerCase().includes(searchTerm)
      )
    );
  }

  // Export immutable data
  toJSON(): MedicalProcedure {
    return { ...this.procedure };
  }

  // Getters for read-only access
  get id(): ProcedureId { return this.procedure.id; }
  get name(): ProcedureName { return this.procedure.name; }
  get category(): ProcedureCategory { return this.procedure.category; }
  get complexity(): ComplexityLevel { return this.procedure.complexity; }
  get icd10Codes(): readonly ICD10Code[] { return this.procedure.icd10Codes; }
  get cptCodes(): readonly CPTCode[] { return this.procedure.cptCodes; }
  get details(): ProcedureDetails { return this.procedure.details; }
  get pricing(): ProcedurePricing { return this.procedure.pricing; }
  get outcome(): ProcedureOutcome { return this.procedure.outcome; }
  get alternativeNames(): readonly string[] { return this.procedure.alternativeNames; }
  get relatedProcedures(): readonly ProcedureId[] { return this.procedure.relatedProcedures; }
  get bodyAreas(): readonly string[] { return this.procedure.bodyAreas; }
  get ageRestrictions() { return this.procedure.ageRestrictions; }
  get genderRestrictions() { return this.procedure.genderRestrictions; }
  get lastUpdated(): Date { return this.procedure.lastUpdated; }
  get extractionMetadata() { return this.procedure.extractionMetadata; }
}

// Utility functions
export const isProcedureValid = (procedure: MedicalProcedure): boolean => {
  return MedicalProcedureSchema.safeParse(procedure).success;
};

export const compareProcedures = (a: MedicalProcedure, b: MedicalProcedure): number => {
  // Compare by complexity first, then by success rate
  const complexityComparison = a.complexity.localeCompare(b.complexity);
  if (complexityComparison !== 0) return complexityComparison;
  
  return b.outcome.successRate - a.outcome.successRate;
};

export const filterProceduresByCategory = (
  procedures: MedicalProcedure[],
  category: ProcedureCategoryType
): MedicalProcedure[] => {
  return procedures.filter(procedure => procedure.category === category);
};

export const filterProceduresByComplexity = (
  procedures: MedicalProcedure[],
  complexity: ComplexityLevel
): MedicalProcedure[] => {
  return procedures.filter(procedure => procedure.complexity === complexity);
};

export const searchProcedures = (
  procedures: MedicalProcedure[],
  searchTerm: string
): MedicalProcedure[] => {
  const term = searchTerm.toLowerCase();
  return procedures.filter(procedure => {
    const aggregate = MedicalProcedureAggregate.create(procedure);
    if (aggregate._tag === "Right") {
      return aggregate.right.matchesSearchTerm(term);
    }
    return false;
  });
};

export const getProceduresByBodyArea = (
  procedures: MedicalProcedure[],
  bodyArea: string
): MedicalProcedure[] => {
  const area = bodyArea.toLowerCase();
  return procedures.filter(procedure => 
    procedure.bodyAreas.some(ba => ba.toLowerCase().includes(area))
  );
};

export const getProceduresByPriceRange = (
  procedures: MedicalProcedure[],
  minPrice: number,
  maxPrice: number
): MedicalProcedure[] => {
  return procedures.filter(procedure => {
    const averagePrice = procedure.pricing.priceRange.average;
    return averagePrice.value >= minPrice && averagePrice.value <= maxPrice;
  });
};

// Common medical aesthetics procedures
export const COMMON_PROCEDURES = {
  BOTOX: "Botox Injection",
  FILLER: "Dermal Filler",
  LIPOSUCTION: "Liposuction",
  FACELIFT: "Facelift",
  RHINOPLASTY: "Rhinoplasty",
  BREAST_AUGMENTATION: "Breast Augmentation",
  LASER_RESURFACING: "Laser Skin Resurfacing",
  CHEMICAL_PEEL: "Chemical Peel",
  MICRODERMABRASION: "Microdermabrasion",
  COOLSCULPTING: "CoolSculpting"
} as const;

export const getProcedureByName = (
  procedures: MedicalProcedure[],
  name: string
): MedicalProcedure | null => {
  return procedures.find(procedure => 
    procedure.name.toLowerCase() === name.toLowerCase() ||
    procedure.alternativeNames.some(alt => 
      alt.toLowerCase() === name.toLowerCase()
    )
  ) || null;
};
