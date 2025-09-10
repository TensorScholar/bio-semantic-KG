/**
 * Price Value Object - Immutable Monetary Value
 * 
 * Represents a monetary value with currency and validation.
 * Implements value object pattern with immutability guarantees.
 * 
 * @file price.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

import { z } from "zod";
import { Either, left, right } from "fp-ts/Either";
import { Branded } from "../../shared/types/branded.ts";

// Branded types for type safety
export type Currency = Branded<string, "Currency">;
export type PriceValue = Branded<number, "PriceValue">;

// Supported currencies with their symbols and decimal places
export const SUPPORTED_CURRENCIES = {
  USD: { symbol: "$", decimals: 2, name: "US Dollar" },
  EUR: { symbol: "€", decimals: 2, name: "Euro" },
  GBP: { symbol: "£", decimals: 2, name: "British Pound" },
  IRR: { symbol: "﷼", decimals: 0, name: "Iranian Rial" },
  AED: { symbol: "د.إ", decimals: 2, name: "UAE Dirham" },
  SAR: { symbol: "﷼", decimals: 2, name: "Saudi Riyal" },
  CAD: { symbol: "C$", decimals: 2, name: "Canadian Dollar" },
  AUD: { symbol: "A$", decimals: 2, name: "Australian Dollar" }
} as const;

export type SupportedCurrency = keyof typeof SUPPORTED_CURRENCIES;

// Validation schemas
const CurrencySchema = z.enum([
  "USD", "EUR", "GBP", "IRR", "AED", "SAR", "CAD", "AUD"
]).brand<"Currency">();

const PriceValueSchema = z.number()
  .positive()
  .finite()
  .brand<"PriceValue">();

// Price range for procedures
export interface PriceRange {
  readonly min: PriceValue;
  readonly max: PriceValue;
  readonly currency: Currency;
  readonly isEstimate: boolean;
  readonly lastUpdated: Date;
}

const PriceRangeSchema = z.object({
  min: PriceValueSchema,
  max: PriceValueSchema,
  currency: CurrencySchema,
  isEstimate: z.boolean(),
  lastUpdated: z.date()
}).refine(
  (data) => data.max >= data.min,
  { message: "Maximum price must be greater than or equal to minimum price" }
);

// Main Price value object
export interface Price {
  readonly value: PriceValue;
  readonly currency: Currency;
  readonly formatted: string;
  readonly createdAt: Date;
}

const PriceSchema = z.object({
  value: PriceValueSchema,
  currency: CurrencySchema,
  formatted: z.string(),
  createdAt: z.date()
});

// Domain errors
export class InvalidPriceError extends Error {
  constructor(message: string, public readonly field: string) {
    super(message);
    this.name = "InvalidPriceError";
  }
}

export class CurrencyMismatchError extends Error {
  constructor(currency1: Currency, currency2: Currency) {
    super(`Cannot perform operation between ${currency1} and ${currency2}`);
    this.name = "CurrencyMismatchError";
  }
}

export class UnsupportedCurrencyError extends Error {
  constructor(currency: string) {
    super(`Currency '${currency}' is not supported`);
    this.name = "UnsupportedCurrencyError";
  }
}

// Factory functions
export const createCurrency = (currency: string): Either<UnsupportedCurrencyError, Currency> => {
  try {
    const result = CurrencySchema.parse(currency);
    return right(result);
  } catch (error) {
    return left(new UnsupportedCurrencyError(currency));
  }
};

export const createPriceValue = (value: number): Either<InvalidPriceError, PriceValue> => {
  try {
    const result = PriceValueSchema.parse(value);
    return right(result);
  } catch (error) {
    return left(new InvalidPriceError("Price value must be a positive number", "value"));
  }
};

// Main Price class with business logic
export class Price {
  private constructor(
    private readonly _value: PriceValue,
    private readonly _currency: Currency,
    private readonly _createdAt: Date = new Date()
  ) {}

  static create(
    value: number,
    currency: string
  ): Either<InvalidPriceError | UnsupportedCurrencyError, Price> {
    const currencyResult = createCurrency(currency);
    if (currencyResult._tag === "Left") {
      return left(currencyResult.left);
    }

    const valueResult = createPriceValue(value);
    if (valueResult._tag === "Left") {
      return left(valueResult.left);
    }

    const price = new Price(valueResult.right, currencyResult.right);
    return right(price);
  }

  static fromString(priceString: string): Either<InvalidPriceError, Price> {
    // Parse various price formats: "$100", "100 USD", "€50.00", etc.
    const currencyRegex = /([A-Z]{3}|\$|€|£|د\.إ|﷼|C\$|A\$)/;
    const numberRegex = /[\d,]+\.?\d*/;
    
    const currencyMatch = priceString.match(currencyRegex);
    const numberMatch = priceString.match(numberRegex);
    
    if (!currencyMatch || !numberMatch) {
      return left(new InvalidPriceError("Invalid price format", "priceString"));
    }

    let currency: string;
    const currencySymbol = currencyMatch[1];
    
    // Map symbols to currency codes
    switch (currencySymbol) {
      case "$": currency = "USD"; break;
      case "€": currency = "EUR"; break;
      case "£": currency = "GBP"; break;
      case "د.إ": currency = "AED"; break;
      case "﷼": currency = "IRR"; break;
      case "C$": currency = "CAD"; break;
      case "A$": currency = "AUD"; break;
      default: currency = currencySymbol; break;
    }

    const value = parseFloat(numberMatch[0].replace(/,/g, ""));
    return Price.create(value, currency);
  }

  // Getters
  get value(): PriceValue { return this._value; }
  get currency(): Currency { return this._currency; }
  get createdAt(): Date { return this._createdAt; }

  // Formatting
  get formatted(): string {
    const currencyInfo = SUPPORTED_CURRENCIES[this._currency];
    const formattedValue = this._value.toFixed(currencyInfo.decimals);
    return `${currencyInfo.symbol}${formattedValue}`;
  }

  get formattedWithCurrency(): string {
    const currencyInfo = SUPPORTED_CURRENCIES[this._currency];
    const formattedValue = this._value.toFixed(currencyInfo.decimals);
    return `${formattedValue} ${this._currency}`;
  }

  // Arithmetic operations
  add(other: Price): Either<CurrencyMismatchError, Price> {
    if (this._currency !== other._currency) {
      return left(new CurrencyMismatchError(this._currency, other._currency));
    }

    const newValue = this._value + other._value;
    return Price.create(newValue, this._currency);
  }

  subtract(other: Price): Either<CurrencyMismatchError, Price> {
    if (this._currency !== other._currency) {
      return left(new CurrencyMismatchError(this._currency, other._currency));
    }

    const newValue = this._value - other._value;
    if (newValue <= 0) {
      return left(new InvalidPriceError("Resulting price must be positive", "subtract"));
    }

    return Price.create(newValue, this._currency);
  }

  multiply(factor: number): Either<InvalidPriceError, Price> {
    if (factor <= 0) {
      return left(new InvalidPriceError("Multiplication factor must be positive", "multiply"));
    }

    const newValue = this._value * factor;
    return Price.create(newValue, this._currency);
  }

  divide(divisor: number): Either<InvalidPriceError, Price> {
    if (divisor <= 0) {
      return left(new InvalidPriceError("Division divisor must be positive", "divide"));
    }

    const newValue = this._value / divisor;
    return Price.create(newValue, this._currency);
  }

  // Comparison operations
  equals(other: Price): boolean {
    return this._value === other._value && this._currency === other._currency;
  }

  isGreaterThan(other: Price): Either<CurrencyMismatchError, boolean> {
    if (this._currency !== other._currency) {
      return left(new CurrencyMismatchError(this._currency, other._currency));
    }
    return right(this._value > other._value);
  }

  isLessThan(other: Price): Either<CurrencyMismatchError, boolean> {
    if (this._currency !== other._currency) {
      return left(new CurrencyMismatchError(this._currency, other._currency));
    }
    return right(this._value < other._value);
  }

  // Currency conversion (simplified - would need real exchange rates)
  convertTo(targetCurrency: Currency): Either<UnsupportedCurrencyError, Price> {
    // This is a simplified conversion - in reality, you'd use real exchange rates
    const exchangeRates: Record<string, Record<string, number>> = {
      USD: { EUR: 0.85, GBP: 0.73, IRR: 42000, AED: 3.67, SAR: 3.75, CAD: 1.25, AUD: 1.35 },
      EUR: { USD: 1.18, GBP: 0.86, IRR: 49500, AED: 4.32, SAR: 4.41, CAD: 1.47, AUD: 1.59 },
      GBP: { USD: 1.37, EUR: 1.16, IRR: 57500, AED: 5.02, SAR: 5.13, CAD: 1.71, AUD: 1.85 },
      IRR: { USD: 0.000024, EUR: 0.000020, GBP: 0.000017, AED: 0.000087, SAR: 0.000089, CAD: 0.000030, AUD: 0.000032 },
      AED: { USD: 0.27, EUR: 0.23, GBP: 0.20, IRR: 11500, SAR: 1.02, CAD: 0.34, AUD: 0.37 },
      SAR: { USD: 0.27, EUR: 0.23, GBP: 0.19, IRR: 11200, AED: 0.98, CAD: 0.33, AUD: 0.36 },
      CAD: { USD: 0.80, EUR: 0.68, GBP: 0.58, IRR: 33600, AED: 2.93, SAR: 3.00, AUD: 1.08 },
      AUD: { USD: 0.74, EUR: 0.63, GBP: 0.54, IRR: 31000, AED: 2.70, SAR: 2.76, CAD: 0.93 }
    };

    if (this._currency === targetCurrency) {
      return right(this);
    }

    const rate = exchangeRates[this._currency]?.[targetCurrency];
    if (!rate) {
      return left(new UnsupportedCurrencyError(`Conversion from ${this._currency} to ${targetCurrency} not supported`));
    }

    const convertedValue = this._value * rate;
    return Price.create(convertedValue, targetCurrency);
  }

  // Validation
  isValid(): boolean {
    return PriceSchema.safeParse({
      value: this._value,
      currency: this._currency,
      formatted: this.formatted,
      createdAt: this._createdAt
    }).success;
  }

  // Serialization
  toJSON(): { value: number; currency: string; formatted: string; createdAt: string } {
    return {
      value: this._value,
      currency: this._currency,
      formatted: this.formatted,
      createdAt: this._createdAt.toISOString()
    };
  }

  static fromJSON(data: { value: number; currency: string; createdAt: string }): Either<InvalidPriceError | UnsupportedCurrencyError, Price> {
    const currencyResult = createCurrency(data.currency);
    if (currencyResult._tag === "Left") {
      return left(currencyResult.left);
    }

    const valueResult = createPriceValue(data.value);
    if (valueResult._tag === "Left") {
      return left(valueResult.left);
    }

    const price = new Price(valueResult.right, currencyResult.right, new Date(data.createdAt));
    return right(price);
  }
}

// PriceRange factory and utilities
export class PriceRange {
  private constructor(
    private readonly _min: PriceValue,
    private readonly _max: PriceValue,
    private readonly _currency: Currency,
    private readonly _isEstimate: boolean,
    private readonly _lastUpdated: Date
  ) {}

  static create(
    min: number,
    max: number,
    currency: string,
    isEstimate: boolean = false
  ): Either<InvalidPriceError | UnsupportedCurrencyError, PriceRange> {
    const currencyResult = createCurrency(currency);
    if (currencyResult._tag === "Left") {
      return left(currencyResult.left);
    }

    const minResult = createPriceValue(min);
    if (minResult._tag === "Left") {
      return left(minResult.left);
    }

    const maxResult = createPriceValue(max);
    if (maxResult._tag === "Left") {
      return left(maxResult.left);
    }

    if (maxResult.right < minResult.right) {
      return left(new InvalidPriceError("Maximum price must be greater than or equal to minimum price", "range"));
    }

    const range = new PriceRange(
      minResult.right,
      maxResult.right,
      currencyResult.right,
      isEstimate,
      new Date()
    );

    return right(range);
  }

  // Getters
  get min(): PriceValue { return this._min; }
  get max(): PriceValue { return this._max; }
  get currency(): Currency { return this._currency; }
  get isEstimate(): boolean { return this._isEstimate; }
  get lastUpdated(): Date { return this._lastUpdated; }

  // Calculations
  get average(): Price {
    const avgValue = (this._min + this._max) / 2;
    return Price.create(avgValue, this._currency).getOrElse(
      Price.create(0, this._currency).getOrElse(null as any)
    );
  }

  get range(): number {
    return this._max - this._min;
  }

  get formatted(): string {
    const currencyInfo = SUPPORTED_CURRENCIES[this._currency];
    const minFormatted = this._min.toFixed(currencyInfo.decimals);
    const maxFormatted = this._max.toFixed(currencyInfo.decimals);
    return `${currencyInfo.symbol}${minFormatted} - ${currencyInfo.symbol}${maxFormatted}`;
  }

  // Validation
  contains(price: Price): Either<CurrencyMismatchError, boolean> {
    if (this._currency !== price.currency) {
      return left(new CurrencyMismatchError(this._currency, price.currency));
    }
    return right(price.value >= this._min && price.value <= this._max);
  }

  // Serialization
  toJSON(): { min: number; max: number; currency: string; isEstimate: boolean; lastUpdated: string } {
    return {
      min: this._min,
      max: this._max,
      currency: this._currency,
      isEstimate: this._isEstimate,
      lastUpdated: this._lastUpdated.toISOString()
    };
  }
}

// Utility functions
export const formatPrice = (value: number, currency: string): string => {
  const price = Price.create(value, currency);
  if (price._tag === "Right") {
    return price.right.formatted;
  }
  return `${value} ${currency}`;
};

export const parsePriceFromText = (text: string): Price[] => {
  const prices: Price[] = [];
  const priceRegex = /([A-Z]{3}|\$|€|£|د\.إ|﷼|C\$|A\$)\s*([\d,]+\.?\d*)/g;
  
  let match;
  while ((match = priceRegex.exec(text)) !== null) {
    const priceString = match[0];
    const price = Price.fromString(priceString);
    if (price._tag === "Right") {
      prices.push(price.right);
    }
  }
  
  return prices;
};

export const sortPrices = (prices: Price[]): Price[] => {
  return prices.sort((a, b) => {
    // First sort by currency, then by value
    if (a.currency !== b.currency) {
      return a.currency.localeCompare(b.currency);
    }
    return a.value - b.value;
  });
};
