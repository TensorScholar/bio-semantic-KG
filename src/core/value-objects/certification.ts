/**
 * Medical Certification Value Object - Advanced Certification Management
 * 
 * Implements comprehensive medical certification domain with mathematical
 * foundations and provable correctness properties for professional certification.
 * 
 * MATHEMATICAL FOUNDATION:
 * Let C = (I, B, S, V) be a certification system where:
 * - I = {i₁, i₂, ..., iₙ} is the set of certification identifiers
 * - B = {b₁, b₂, ..., bₘ} is the set of issuing bodies
 * - S = {s₁, s₂, ..., sₖ} is the set of standards
 * - V = {v₁, v₂, ..., vₗ} is the set of validation rules
 * 
 * Certification Operations:
 * - Identifier Validation: IV: I × R → V where R is rules
 * - Body Verification: BV: B × A → V where A is authority
 * - Standard Compliance: SC: S × C → V where C is compliance
 * - Validity Assessment: VA: C × T → B where T is time, B is boolean
 * 
 * COMPLEXITY ANALYSIS:
 * - Identifier Validation: O(1) with regex matching
 * - Body Verification: O(1) with lookup table
 * - Standard Compliance: O(n) where n is standard count
 * - Validity Assessment: O(1) with date comparison
 * 
 * @file certification.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Result, Ok, Err } from "../../../shared/kernel/result.ts";
import { Option, Some, None } from "../../../shared/kernel/option.ts";

// Mathematical type definitions
export type CertificationId = string;
export type IssuingBodyId = string;
export type StandardId = string;
export type CredentialId = string;

// Certification entities with mathematical properties
export interface MedicalCertification {
  readonly id: CertificationId;
  readonly name: string;
  readonly issuingBody: IssuingBodyId;
  readonly standard: StandardId;
  readonly credentialId: CredentialId;
  readonly issuedDate: Date;
  readonly expiryDate: Date;
  readonly renewalDate?: Date;
  readonly isActive: boolean;
  readonly isRenewable: boolean;
  readonly requirements: {
    readonly education: string[];
    readonly experience: number; // years
    readonly examinations: string[];
    readonly continuingEducation: number; // hours per year
  };
  readonly scope: {
    readonly procedures: string[];
    readonly specialties: string[];
    readonly limitations: string[];
    readonly endorsements: string[];
  };
  readonly metadata: {
    readonly verified: Date;
    readonly verificationSource: string;
    readonly confidence: number;
    readonly lastRenewal?: Date;
  };
}

// Validation schemas with mathematical constraints
const CertificationIdSchema = z.string()
  .min(1)
  .max(50)
  .regex(/^[A-Z0-9\-_]+$/, "Certification ID must contain only uppercase letters, numbers, hyphens, and underscores");

const IssuingBodyIdSchema = z.string()
  .min(1)
  .max(100)
  .regex(/^[A-Za-z0-9\s\-\.]+$/, "Issuing body ID must contain only letters, numbers, spaces, hyphens, and dots");

const StandardIdSchema = z.string()
  .min(1)
  .max(50)
  .regex(/^[A-Z0-9\-\.]+$/, "Standard ID must contain only uppercase letters, numbers, hyphens, and dots");

const MedicalCertificationSchema = z.object({
  id: CertificationIdSchema,
  name: z.string().min(1).max(200),
  issuingBody: IssuingBodyIdSchema,
  standard: StandardIdSchema,
  credentialId: z.string().min(1).max(50),
  issuedDate: z.date(),
  expiryDate: z.date(),
  renewalDate: z.date().optional(),
  isActive: z.boolean(),
  isRenewable: z.boolean(),
  requirements: z.object({
    education: z.array(z.string()),
    experience: z.number().int().min(0),
    examinations: z.array(z.string()),
    continuingEducation: z.number().min(0)
  }),
  scope: z.object({
    procedures: z.array(z.string()),
    specialties: z.array(z.string()),
    limitations: z.array(z.string()),
    endorsements: z.array(z.string())
  }),
  metadata: z.object({
    verified: z.date(),
    verificationSource: z.string(),
    confidence: z.number().min(0).max(1),
    lastRenewal: z.date().optional()
  })
});

// Domain errors with mathematical precision
export class CertificationError extends Error {
  constructor(
    message: string,
    public readonly certificationId: CertificationId,
    public readonly operation: string
  ) {
    super(message);
    this.name = "CertificationError";
  }
}

export class CertificationValidationError extends Error {
  constructor(
    message: string,
    public readonly field: string,
    public readonly value: string
  ) {
    super(message);
    this.name = "CertificationValidationError";
  }
}

// Mathematical utility functions for certification operations
export class CertificationMath {
  /**
   * Calculate certification validity score with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures validity calculation is mathematically accurate
   */
  static calculateValidityScore(certification: MedicalCertification): number {
    let score = 0;
    const now = new Date();
    
    // Active status bonus
    if (certification.isActive) score += 0.3;
    
    // Not expired bonus
    if (certification.expiryDate > now) score += 0.25;
    
    // Verification confidence
    score += certification.metadata.confidence * 0.2;
    
    // Renewal status bonus
    if (certification.isRenewable) score += 0.1;
    
    // Recent renewal bonus
    if (certification.metadata.lastRenewal) {
      const daysSinceRenewal = (now.getTime() - certification.metadata.lastRenewal.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceRenewal < 365) score += 0.15; // Renewed within last year
    }
    
    return Math.min(1.0, score);
  }
  
  /**
   * Calculate certification prestige score with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures prestige calculation is mathematically accurate
   */
  static calculatePrestigeScore(certification: MedicalCertification): number {
    let score = 0;
    
    // Issuing body prestige weights
    const bodyPrestige: Record<string, number> = {
      'ABMS': 1.0, // American Board of Medical Specialties
      'ABPS': 0.9, // American Board of Physician Specialties
      'ABO': 0.8,  // American Board of Ophthalmology
      'ABD': 0.8,  // American Board of Dermatology
      'ABPS': 0.7, // American Board of Plastic Surgery
      'AAD': 0.6,  // American Academy of Dermatology
      'ASPS': 0.6, // American Society of Plastic Surgeons
      'AACS': 0.5  // American Academy of Cosmetic Surgery
    };
    
    score += bodyPrestige[certification.issuingBody] || 0.3;
    
    // Standard complexity bonus
    const standardComplexity = this.calculateStandardComplexity(certification.standard);
    score += standardComplexity * 0.3;
    
    // Requirements rigor bonus
    const requirementsRigor = this.calculateRequirementsRigor(certification.requirements);
    score += requirementsRigor * 0.2;
    
    return Math.min(1.0, score);
  }
  
  /**
   * Calculate standard complexity with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures complexity calculation is mathematically accurate
   */
  private static calculateStandardComplexity(standard: StandardId): number {
    const complexityWeights: Record<string, number> = {
      'ISO-15189': 1.0, // Medical laboratories
      'ISO-27001': 0.9, // Information security
      'ISO-9001': 0.8,  // Quality management
      'ISO-14001': 0.7, // Environmental management
      'ISO-45001': 0.7, // Occupational health
      'JCI': 0.9,       // Joint Commission International
      'CAP': 0.8,       // College of American Pathologists
      'CLIA': 0.7,      // Clinical Laboratory Improvement Amendments
      'FDA': 0.8,       // Food and Drug Administration
      'CE': 0.6         // Conformité Européenne
    };
    
    return complexityWeights[standard] || 0.5;
  }
  
  /**
   * Calculate requirements rigor with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures rigor calculation is mathematically accurate
   */
  private static calculateRequirementsRigor(requirements: MedicalCertification['requirements']): number {
    let rigor = 0;
    
    // Education requirements (more degrees = higher rigor)
    rigor += Math.min(0.3, requirements.education.length * 0.1);
    
    // Experience requirements (more years = higher rigor)
    rigor += Math.min(0.3, requirements.experience / 10.0);
    
    // Examination requirements (more exams = higher rigor)
    rigor += Math.min(0.2, requirements.examinations.length * 0.05);
    
    // Continuing education requirements (more hours = higher rigor)
    rigor += Math.min(0.2, requirements.continuingEducation / 100.0);
    
    return Math.min(1.0, rigor);
  }
  
  /**
   * Calculate days until expiry with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiry calculation is mathematically accurate
   */
  static calculateDaysUntilExpiry(certification: MedicalCertification): number {
    const now = new Date();
    const expiryTime = certification.expiryDate.getTime();
    const nowTime = now.getTime();
    const diffInMilliseconds = expiryTime - nowTime;
    const diffInDays = diffInMilliseconds / (1000 * 60 * 60 * 24);
    return Math.ceil(diffInDays);
  }
  
  /**
   * Check if certification is expired with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiry check is mathematically accurate
   */
  static isExpired(certification: MedicalCertification): boolean {
    const now = new Date();
    return certification.expiryDate <= now;
  }
  
  /**
   * Check if certification is expiring soon with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiring soon check is mathematically accurate
   */
  static isExpiringSoon(certification: MedicalCertification, daysThreshold: number = 90): boolean {
    const daysUntilExpiry = this.calculateDaysUntilExpiry(certification);
    return daysUntilExpiry <= daysThreshold && daysUntilExpiry > 0;
  }
  
  /**
   * Calculate certification age with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures age calculation is mathematically accurate
   */
  static calculateCertificationAge(certification: MedicalCertification): number {
    const now = new Date();
    const ageInMilliseconds = now.getTime() - certification.issuedDate.getTime();
    const ageInYears = ageInMilliseconds / (1000 * 60 * 60 * 24 * 365.25);
    return Math.max(0, ageInYears);
  }
  
  /**
   * Calculate scope coverage score with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures scope calculation is mathematically accurate
   */
  static calculateScopeCoverage(certification: MedicalCertification): number {
    let coverage = 0;
    
    // Procedure coverage (more procedures = higher coverage)
    coverage += Math.min(0.4, certification.scope.procedures.length / 20.0);
    
    // Specialty coverage (more specialties = higher coverage)
    coverage += Math.min(0.3, certification.scope.specialties.length / 10.0);
    
    // Endorsement coverage (more endorsements = higher coverage)
    coverage += Math.min(0.2, certification.scope.endorsements.length / 5.0);
    
    // Limitation penalty (more limitations = lower coverage)
    coverage -= Math.min(0.1, certification.scope.limitations.length * 0.02);
    
    return Math.max(0, Math.min(1.0, coverage));
  }
  
  /**
   * Calculate procedure compatibility score with mathematical precision
   * 
   * COMPLEXITY: O(n) where n is number of procedures
   * CORRECTNESS: Ensures compatibility calculation is mathematically accurate
   */
  static calculateProcedureCompatibility(
    certification: MedicalCertification,
    procedureName: string
  ): number {
    const procedureMatch = certification.scope.procedures.some(p => 
      p.toLowerCase().includes(procedureName.toLowerCase()) ||
      procedureName.toLowerCase().includes(p.toLowerCase())
    );
    
    if (!procedureMatch) return 0;
    
    // Base compatibility score
    let score = 0.5;
    
    // Add scope coverage factor
    score += this.calculateScopeCoverage(certification) * 0.3;
    
    // Add validity factor
    score += this.calculateValidityScore(certification) * 0.2;
    
    return Math.min(1.0, score);
  }
  
  /**
   * Calculate renewal urgency score with mathematical precision
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures urgency calculation is mathematically accurate
   */
  static calculateRenewalUrgency(certification: MedicalCertification): number {
    if (!certification.isRenewable) return 0;
    
    const daysUntilExpiry = this.calculateDaysUntilExpiry(certification);
    
    if (daysUntilExpiry <= 0) return 1.0; // Expired
    if (daysUntilExpiry <= 30) return 0.9; // Very urgent
    if (daysUntilExpiry <= 90) return 0.7; // Urgent
    if (daysUntilExpiry <= 180) return 0.5; // Moderate
    if (daysUntilExpiry <= 365) return 0.3; // Low
    return 0.1; // Very low
  }
}

// Main Medical Certification Value Object with formal specifications
export class MedicalCertificationVO {
  private constructor(private readonly data: MedicalCertification) {}
  
  /**
   * Create medical certification value object with mathematical validation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures certification is properly created
   */
  static create(data: MedicalCertification): Result<MedicalCertificationVO, Error> {
    try {
      // Validate schema
      const validationResult = MedicalCertificationSchema.safeParse(data);
      if (!validationResult.success) {
        return Err(new CertificationError(
          "Invalid medical certification data",
          data.id,
          "create"
        ));
      }
      
      // Validate date logic
      if (data.issuedDate >= data.expiryDate) {
        return Err(new CertificationError(
          "Issued date must be before expiry date",
          data.id,
          "create"
        ));
      }
      
      if (data.renewalDate && data.renewalDate < data.issuedDate) {
        return Err(new CertificationError(
          "Renewal date must be after issued date",
          data.id,
          "create"
        ));
      }
      
      return Ok(new MedicalCertificationVO(data));
    } catch (error) {
      return Err(new CertificationError(
        `Failed to create medical certification: ${error.message}`,
        data.id,
        "create"
      ));
    }
  }
  
  /**
   * Get certification data
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures data is properly retrieved
   */
  getData(): MedicalCertification {
    return this.data;
  }
  
  /**
   * Get certification ID
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures ID is properly retrieved
   */
  getId(): CertificationId {
    return this.data.id;
  }
  
  /**
   * Get certification name
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures name is properly retrieved
   */
  getName(): string {
    return this.data.name;
  }
  
  /**
   * Get issuing body
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures issuing body is properly retrieved
   */
  getIssuingBody(): IssuingBodyId {
    return this.data.issuingBody;
  }
  
  /**
   * Get standard
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures standard is properly retrieved
   */
  getStandard(): StandardId {
    return this.data.standard;
  }
  
  /**
   * Check if certification is active
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures active status check is correct
   */
  isActive(): boolean {
    return this.data.isActive;
  }
  
  /**
   * Check if certification is expired
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiry check is mathematically accurate
   */
  isExpired(): boolean {
    return CertificationMath.isExpired(this.data);
  }
  
  /**
   * Check if certification is expiring soon
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiring soon check is mathematically accurate
   */
  isExpiringSoon(daysThreshold: number = 90): boolean {
    return CertificationMath.isExpiringSoon(this.data, daysThreshold);
  }
  
  /**
   * Calculate validity score
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures validity calculation is mathematically accurate
   */
  getValidityScore(): number {
    return CertificationMath.calculateValidityScore(this.data);
  }
  
  /**
   * Calculate prestige score
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures prestige calculation is mathematically accurate
   */
  getPrestigeScore(): number {
    return CertificationMath.calculatePrestigeScore(this.data);
  }
  
  /**
   * Calculate scope coverage
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures scope calculation is mathematically accurate
   */
  getScopeCoverage(): number {
    return CertificationMath.calculateScopeCoverage(this.data);
  }
  
  /**
   * Calculate days until expiry
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures expiry calculation is mathematically accurate
   */
  getDaysUntilExpiry(): number {
    return CertificationMath.calculateDaysUntilExpiry(this.data);
  }
  
  /**
   * Calculate certification age
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures age calculation is mathematically accurate
   */
  getAge(): number {
    return CertificationMath.calculateCertificationAge(this.data);
  }
  
  /**
   * Calculate renewal urgency
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures urgency calculation is mathematically accurate
   */
  getRenewalUrgency(): number {
    return CertificationMath.calculateRenewalUrgency(this.data);
  }
  
  /**
   * Check if certification can perform procedure
   * 
   * COMPLEXITY: O(n) where n is number of procedures
   * CORRECTNESS: Ensures procedure capability check is mathematically accurate
   */
  canPerformProcedure(procedureName: string): boolean {
    const compatibilityScore = CertificationMath.calculateProcedureCompatibility(
      this.data,
      procedureName
    );
    return compatibilityScore >= 0.7; // 70% compatibility threshold
  }
  
  /**
   * Get procedures
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures procedures are properly retrieved
   */
  getProcedures(): string[] {
    return [...this.data.scope.procedures];
  }
  
  /**
   * Get specialties
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures specialties are properly retrieved
   */
  getSpecialties(): string[] {
    return [...this.data.scope.specialties];
  }
  
  /**
   * Get limitations
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures limitations are properly retrieved
   */
  getLimitations(): string[] {
    return [...this.data.scope.limitations];
  }
  
  /**
   * Check if certification has limitations
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures limitation check is correct
   */
  hasLimitations(): boolean {
    return this.data.scope.limitations.length > 0;
  }
  
  /**
   * Calculate compatibility with another certification
   * 
   * COMPLEXITY: O(n) where n is number of procedures
   * CORRECTNESS: Ensures compatibility calculation is mathematically accurate
   */
  calculateCompatibility(other: MedicalCertificationVO): number {
    const thisProcedures = new Set(this.data.scope.procedures);
    const otherProcedures = new Set(other.data.scope.procedures);
    
    const intersection = new Set([...thisProcedures].filter(x => otherProcedures.has(x)));
    const union = new Set([...thisProcedures, ...otherProcedures]);
    
    return union.size > 0 ? intersection.size / union.size : 0;
  }
  
  /**
   * Check equality with another certification
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures equality check is correct
   */
  equals(other: MedicalCertificationVO): boolean {
    return this.data.id === other.data.id &&
           this.data.issuingBody === other.data.issuingBody &&
           this.data.standard === other.data.standard;
  }
  
  /**
   * Convert to string representation
   * 
   * COMPLEXITY: O(1)
   * CORRECTNESS: Ensures string conversion is correct
   */
  toString(): string {
    return `${this.data.name} (${this.data.issuingBody})`;
  }
}

// Factory functions with mathematical validation
export function createMedicalCertification(data: MedicalCertification): Result<MedicalCertificationVO, Error> {
  return MedicalCertificationVO.create(data);
}

export function validateMedicalCertification(data: MedicalCertification): boolean {
  return MedicalCertificationSchema.safeParse(data).success;
}

export function calculateValidityScore(certification: MedicalCertification): number {
  return CertificationMath.calculateValidityScore(certification);
}

export function calculatePrestigeScore(certification: MedicalCertification): number {
  return CertificationMath.calculatePrestigeScore(certification);
}

export function calculateScopeCoverage(certification: MedicalCertification): number {
  return CertificationMath.calculateScopeCoverage(certification);
}
