/**
 * URL Value Object - Immutable URL with Validation
 * 
 * Represents a validated URL with domain-specific validation rules.
 * Implements value object pattern with security considerations.
 * 
 * @file url.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Branded } from "../../shared/types/branded.ts";

// Branded types for type safety
export type URLString = Branded<string, "URLString">;
export type Domain = Branded<string, "Domain">;
export type Protocol = Branded<string, "Protocol">;

// Supported protocols
export const SUPPORTED_PROTOCOLS = ["http:", "https:", "ftp:", "ftps:"] as const;
export type SupportedProtocol = typeof SUPPORTED_PROTOCOLS[number];

// URL components
export interface URLComponents {
  readonly protocol: Protocol;
  readonly hostname: Domain;
  readonly port?: number;
  readonly pathname: string;
  readonly search?: string;
  readonly hash?: string;
  readonly username?: string;
  readonly password?: string;
}

// Validation schemas
const URLStringSchema = z.string().url().brand<"URLString">();
const DomainSchema = z.string()
  .min(1)
  .max(253)
  .regex(/^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/)
  .brand<"Domain">();

const ProtocolSchema = z.enum(SUPPORTED_PROTOCOLS).brand<"Protocol">();

const URLComponentsSchema = z.object({
  protocol: ProtocolSchema,
  hostname: DomainSchema,
  port: z.number().int().min(1).max(65535).optional(),
  pathname: z.string(),
  search: z.string().optional(),
  hash: z.string().optional(),
  username: z.string().optional(),
  password: z.string().optional()
});

// Main URL value object
export interface URL {
  readonly value: URLString;
  readonly components: URLComponents;
  readonly isValid: boolean;
  readonly isSecure: boolean;
  readonly domain: Domain;
  readonly createdAt: Date;
}

const URLSchema = z.object({
  value: URLStringSchema,
  components: URLComponentsSchema,
  isValid: z.boolean(),
  isSecure: z.boolean(),
  domain: DomainSchema,
  createdAt: z.date()
});

// Domain errors
export class InvalidURLError extends Error {
  constructor(message: string, public readonly field: string) {
    super(message);
    this.name = "InvalidURLError";
  }
}

export class UnsupportedProtocolError extends Error {
  constructor(protocol: string) {
    super(`Protocol '${protocol}' is not supported`);
    this.name = "UnsupportedProtocolError";
  }
}

export class MalformedDomainError extends Error {
  constructor(domain: string) {
    super(`Domain '${domain}' is malformed`);
    this.name = "MalformedDomainError";
  }
}

export class SecurityViolationError extends Error {
  constructor(message: string) {
    super(`Security violation: ${message}`);
    this.name = "SecurityViolationError";
  }
}

// Factory functions
export const createURLString = (url: string): Either<InvalidURLError, URLString> => {
  try {
    const result = URLStringSchema.parse(url);
    return right(result);
  } catch (error) {
    return left(new InvalidURLError("Invalid URL format", "url"));
  }
};

export const createDomain = (domain: string): Either<MalformedDomainError, Domain> => {
  try {
    const result = DomainSchema.parse(domain);
    return right(result);
  } catch (error) {
    return left(new MalformedDomainError(domain));
  }
};

export const createProtocol = (protocol: string): Either<UnsupportedProtocolError, Protocol> => {
  try {
    const result = ProtocolSchema.parse(protocol);
    return right(result);
  } catch (error) {
    return left(new UnsupportedProtocolError(protocol));
  }
};

// Main URL class with business logic
export class URL {
  private constructor(
    private readonly _value: URLString,
    private readonly _components: URLComponents,
    private readonly _createdAt: Date = new Date()
  ) {}

  static create(urlString: string): Either<InvalidURLError | UnsupportedProtocolError | MalformedDomainError, URL> {
    const urlResult = createURLString(urlString);
    if (urlResult._tag === "Left") {
      return left(urlResult.left);
    }

    try {
      const url = new globalThis.URL(urlString);
      
      const protocolResult = createProtocol(url.protocol);
      if (protocolResult._tag === "Left") {
        return left(protocolResult.left);
      }

      const domainResult = createDomain(url.hostname);
      if (domainResult._tag === "Left") {
        return left(domainResult.left);
      }

      const components: URLComponents = {
        protocol: protocolResult.right,
        hostname: domainResult.right,
        port: url.port ? parseInt(url.port) : undefined,
        pathname: url.pathname,
        search: url.search || undefined,
        hash: url.hash || undefined,
        username: url.username || undefined,
        password: url.password || undefined
      };

      const urlObj = new URL(urlResult.right, components);
      return right(urlObj);
    } catch (error) {
      return left(new InvalidURLError("Failed to parse URL components", "url"));
    }
  }

  static fromComponents(components: URLComponents): Either<InvalidURLError, URL> {
    try {
      const url = new globalThis.URL();
      url.protocol = components.protocol;
      url.hostname = components.hostname;
      if (components.port) url.port = components.port.toString();
      url.pathname = components.pathname;
      if (components.search) url.search = components.search;
      if (components.hash) url.hash = components.hash;
      if (components.username) url.username = components.username;
      if (components.password) url.password = components.password;

      const urlString = url.toString();
      const urlResult = createURLString(urlString);
      if (urlResult._tag === "Left") {
        return left(urlResult.left);
      }

      const urlObj = new URL(urlResult.right, components);
      return right(urlObj);
    } catch (error) {
      return left(new InvalidURLError("Failed to construct URL from components", "components"));
    }
  }

  // Getters
  get value(): URLString { return this._value; }
  get components(): URLComponents { return this._components; }
  get createdAt(): Date { return this._createdAt; }

  // Computed properties
  get isValid(): boolean {
    try {
      new globalThis.URL(this._value);
      return true;
    } catch {
      return false;
    }
  }

  get isSecure(): boolean {
    return this._components.protocol === "https:" || this._components.protocol === "ftps:";
  }

  get domain(): Domain { return this._components.hostname; }

  get isLocalhost(): boolean {
    return this._components.hostname === "localhost" || 
           this._components.hostname === "127.0.0.1" ||
           this._components.hostname.startsWith("192.168.") ||
           this._components.hostname.startsWith("10.") ||
           this._components.hostname.startsWith("172.");
  }

  get isIPAddress(): boolean {
    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    const ipv6Regex = /^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/;
    return ipv4Regex.test(this._components.hostname) || ipv6Regex.test(this._components.hostname);
  }

  get topLevelDomain(): string {
    const parts = this._components.hostname.split(".");
    return parts.length > 1 ? parts[parts.length - 1] : "";
  }

  get subdomain(): string {
    const parts = this._components.hostname.split(".");
    return parts.length > 2 ? parts.slice(0, -2).join(".") : "";
  }

  // Security checks
  isSuspicious(): boolean {
    // Check for suspicious patterns
    const suspiciousPatterns = [
      /bit\.ly/i,
      /tinyurl/i,
      /goo\.gl/i,
      /t\.co/i,
      /short\.link/i,
      /redirect/i,
      /phishing/i,
      /malware/i,
      /virus/i
    ];

    return suspiciousPatterns.some(pattern => pattern.test(this._value));
  }

  hasCredentials(): boolean {
    return !!(this._components.username || this._components.password);
  }

  isHTTPS(): boolean {
    return this._components.protocol === "https:";
  }

  // URL manipulation
  withProtocol(protocol: string): Either<UnsupportedProtocolError, URL> {
    const protocolResult = createProtocol(protocol);
    if (protocolResult._tag === "Left") {
      return left(protocolResult.left);
    }

    const newComponents: URLComponents = {
      ...this._components,
      protocol: protocolResult.right
    };

    return URL.fromComponents(newComponents);
  }

  withPath(path: string): Either<InvalidURLError, URL> {
    const newComponents: URLComponents = {
      ...this._components,
      pathname: path
    };

    return URL.fromComponents(newComponents);
  }

  withQuery(query: Record<string, string>): Either<InvalidURLError, URL> {
    const searchParams = new URLSearchParams(query);
    const newComponents: URLComponents = {
      ...this._components,
      search: searchParams.toString()
    };

    return URL.fromComponents(newComponents);
  }

  // Comparison operations
  equals(other: URL): boolean {
    return this._value === other._value;
  }

  hasSameDomain(other: URL): boolean {
    return this._components.hostname === other._components.hostname;
  }

  hasSameProtocol(other: URL): boolean {
    return this._components.protocol === other._components.protocol;
  }

  // Validation
  isValidForExtraction(): Either<SecurityViolationError, void> {
    if (this.isSuspicious()) {
      return left(new SecurityViolationError("URL appears to be suspicious"));
    }

    if (this.hasCredentials()) {
      return left(new SecurityViolationError("URL contains credentials"));
    }

    if (!this.isHTTPS() && !this.isLocalhost()) {
      return left(new SecurityViolationError("Non-HTTPS URLs are not allowed for extraction"));
    }

    return right(undefined);
  }

  // Serialization
  toJSON(): { 
    value: string; 
    components: URLComponents; 
    isValid: boolean; 
    isSecure: boolean; 
    domain: string; 
    createdAt: string 
  } {
    return {
      value: this._value,
      components: this._components,
      isValid: this.isValid,
      isSecure: this.isSecure,
      domain: this._components.hostname,
      createdAt: this._createdAt.toISOString()
    };
  }

  static fromJSON(data: { 
    value: string; 
    components: URLComponents; 
    createdAt: string 
  }): Either<InvalidURLError | UnsupportedProtocolError | MalformedDomainError, URL> {
    const urlResult = createURLString(data.value);
    if (urlResult._tag === "Left") {
      return left(urlResult.left);
    }

    const url = new URL(urlResult.right, data.components, new Date(data.createdAt));
    return right(url);
  }
}

// URL utilities
export const extractDomain = (url: string): Either<InvalidURLError, Domain> => {
  const urlResult = URL.create(url);
  if (urlResult._tag === "Right") {
    return right(urlResult.right.domain);
  }
  return left(urlResult.left);
};

export const isURL = (value: string): boolean => {
  try {
    new globalThis.URL(value);
    return true;
  } catch {
    return false;
  }
};

export const normalizeURL = (url: string): Either<InvalidURLError, URL> => {
  const urlResult = URL.create(url);
  if (urlResult._tag === "Right") {
    return right(urlResult.right);
  }
  return left(urlResult.left);
};

export const validateURLs = (urls: string[]): { valid: URL[]; invalid: string[] } => {
  const valid: URL[] = [];
  const invalid: string[] = [];

  for (const urlString of urls) {
    const urlResult = URL.create(urlString);
    if (urlResult._tag === "Right") {
      valid.push(urlResult.right);
    } else {
      invalid.push(urlString);
    }
  }

  return { valid, invalid };
};

export const filterSecureURLs = (urls: URL[]): URL[] => {
  return urls.filter(url => url.isSecure);
};

export const groupURLsByDomain = (urls: URL[]): Map<Domain, URL[]> => {
  const groups = new Map<Domain, URL[]>();

  for (const url of urls) {
    const domain = url.domain;
    if (!groups.has(domain)) {
      groups.set(domain, []);
    }
    groups.get(domain)!.push(url);
  }

  return groups;
};

export const sortURLs = (urls: URL[]): URL[] => {
  return urls.sort((a, b) => a.value.localeCompare(b.value));
};

// URL patterns for common medical aesthetics websites
export const MEDICAL_AESTHETICS_PATTERNS = {
  CLINIC_WEBSITES: [
    /clinic/i,
    /medical/i,
    /aesthetic/i,
    /beauty/i,
    /dermatology/i,
    /plastic/i,
    /cosmetic/i
  ],
  SOCIAL_MEDIA: [
    /instagram\.com/i,
    /facebook\.com/i,
    /twitter\.com/i,
    /linkedin\.com/i,
    /youtube\.com/i
  ],
  REVIEW_SITES: [
    /google\.com\/maps/i,
    /yelp\.com/i,
    /healthgrades\.com/i,
    /vitals\.com/i,
    /ratemds\.com/i
  ]
} as const;

export const isMedicalAestheticsURL = (url: URL): boolean => {
  const urlString = url.value.toLowerCase();
  
  return MEDICAL_AESTHETICS_PATTERNS.CLINIC_WEBSITES.some(pattern => 
    pattern.test(urlString)
  );
};

export const isSocialMediaURL = (url: URL): boolean => {
  const urlString = url.value.toLowerCase();
  
  return MEDICAL_AESTHETICS_PATTERNS.SOCIAL_MEDIA.some(pattern => 
    pattern.test(urlString)
  );
};

export const isReviewSiteURL = (url: URL): boolean => {
  const urlString = url.value.toLowerCase();
  
  return MEDICAL_AESTHETICS_PATTERNS.REVIEW_SITES.some(pattern => 
    pattern.test(urlString)
  );
};
