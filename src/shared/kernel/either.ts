/**
 * Either Type - Functional Error Handling
 * 
 * Implements the Either pattern for functional error handling with explicit
 * left and right sides. Provides type-safe error handling with composable operations.
 * 
 * @file either.ts
 * @author Elite Technical Consortium
 * @version 1.0.0
 */

/**
 * Either type representing either Left (error) or Right (success)
 */
export type Either<L, R> = Left<L, R> | Right<L, R>;

/**
 * Left variant of Either representing an error
 */
export class Left<L, R> {
  readonly _tag = "Left" as const;
  readonly _left: L;

  constructor(left: L) {
    this._left = left;
  }

  /**
   * Returns false if this is a Left variant
   */
  isLeft(): this is Left<L, R> {
    return true;
  }

  /**
   * Returns false if this is a Left variant
   */
  isRight(): this is Right<L, R> {
    return false;
  }

  /**
   * Returns the left value
   */
  getLeft(): L {
    return this._left;
  }

  /**
   * Returns the right value (throws for Left)
   */
  getRight(): never {
    throw new Error("Called getRight on Left");
  }

  /**
   * Returns the left value or the provided default
   */
  getLeftOr(defaultValue: L): L {
    return this._left;
  }

  /**
   * Returns the right value or the provided default
   */
  getRightOr(defaultValue: R): R {
    return defaultValue;
  }

  /**
   * Returns the left value or computes it from a function
   */
  getLeftOrElse(fn: (right: R) => L): L {
    return this._left;
  }

  /**
   * Returns the right value or computes it from a function
   */
  getRightOrElse(fn: (left: L) => R): R {
    return fn(this._left);
  }

  /**
   * Maps the left value using the provided function
   */
  mapLeft<M>(fn: (left: L) => M): Either<M, R> {
    return new Left(fn(this._left));
  }

  /**
   * Maps the right value using the provided function
   */
  mapRight<N>(fn: (right: R) => N): Either<L, N> {
    return new Left(this._left);
  }

  /**
   * Maps both sides using the provided functions
   */
  mapBoth<M, N>(
    leftFn: (left: L) => M,
    rightFn: (right: R) => N
  ): Either<M, N> {
    return new Left(leftFn(this._left));
  }

  /**
   * Flat maps the left value using the provided function
   */
  flatMapLeft<M>(fn: (left: L) => Either<M, R>): Either<M, R> {
    return fn(this._left);
  }

  /**
   * Flat maps the right value using the provided function
   */
  flatMapRight<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return new Left(this._left);
  }

  /**
   * Chains another Either operation
   */
  chain<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return new Left(this._left);
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  orElse<N>(fn: (left: L) => Either<L, N>): Either<L, R | N> {
    return fn(this._left);
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  or<N>(other: Either<L, N>): Either<L, R | N> {
    return other;
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  and<N>(other: Either<L, N>): Either<L, N> {
    return new Left(this._left);
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  andThen<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return new Left(this._left);
  }

  /**
   * Converts to Result, mapping Left to Err and Right to Ok
   */
  toResult(): Result<R, L> {
    return new Err(this._left);
  }

  /**
   * Converts to Option, mapping Left to None and Right to Some
   */
  toOption(): Option<R> {
    return new None();
  }

  /**
   * Matches on the Either variant
   */
  match<T>(matchers: {
    left: (left: L) => T;
    right: (right: R) => T;
  }): T {
    return matchers.left(this._left);
  }

  /**
   * Executes a function if this is Left
   */
  ifLeft(fn: (left: L) => void): void {
    fn(this._left);
  }

  /**
   * Executes a function if this is Right
   */
  ifRight(fn: (right: R) => void): void {
    // Do nothing for Left variant
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return `Left(${this._left})`;
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "Left"; _left: L } {
    return { _tag: "Left", _left: this._left };
  }
}

/**
 * Right variant of Either representing success
 */
export class Right<L, R> {
  readonly _tag = "Right" as const;
  readonly _right: R;

  constructor(right: R) {
    this._right = right;
  }

  /**
   * Returns false if this is a Right variant
   */
  isLeft(): this is Left<L, R> {
    return false;
  }

  /**
   * Returns true if this is a Right variant
   */
  isRight(): this is Right<L, R> {
    return true;
  }

  /**
   * Returns the left value (throws for Right)
   */
  getLeft(): never {
    throw new Error("Called getLeft on Right");
  }

  /**
   * Returns the right value
   */
  getRight(): R {
    return this._right;
  }

  /**
   * Returns the left value or the provided default
   */
  getLeftOr(defaultValue: L): L {
    return defaultValue;
  }

  /**
   * Returns the right value or the provided default
   */
  getRightOr(defaultValue: R): R {
    return this._right;
  }

  /**
   * Returns the left value or computes it from a function
   */
  getLeftOrElse(fn: (right: R) => L): L {
    return fn(this._right);
  }

  /**
   * Returns the right value or computes it from a function
   */
  getRightOrElse(fn: (left: L) => R): R {
    return this._right;
  }

  /**
   * Maps the left value using the provided function
   */
  mapLeft<M>(fn: (left: L) => M): Either<M, R> {
    return new Right(this._right);
  }

  /**
   * Maps the right value using the provided function
   */
  mapRight<N>(fn: (right: R) => N): Either<L, N> {
    return new Right(fn(this._right));
  }

  /**
   * Maps both sides using the provided functions
   */
  mapBoth<M, N>(
    leftFn: (left: L) => M,
    rightFn: (right: R) => N
  ): Either<M, N> {
    return new Right(rightFn(this._right));
  }

  /**
   * Flat maps the left value using the provided function
   */
  flatMapLeft<M>(fn: (left: L) => Either<M, R>): Either<M, R> {
    return new Right(this._right);
  }

  /**
   * Flat maps the right value using the provided function
   */
  flatMapRight<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return fn(this._right);
  }

  /**
   * Chains another Either operation
   */
  chain<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return fn(this._right);
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  orElse<N>(fn: (left: L) => Either<L, N>): Either<L, R | N> {
    return this;
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  or<N>(other: Either<L, N>): Either<L, R | N> {
    return this;
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  and<N>(other: Either<L, N>): Either<L, N> {
    return other;
  }

  /**
   * Returns this Either if Right, otherwise returns the provided Either
   */
  andThen<N>(fn: (right: R) => Either<L, N>): Either<L, N> {
    return fn(this._right);
  }

  /**
   * Converts to Result, mapping Left to Err and Right to Ok
   */
  toResult(): Result<R, L> {
    return new Ok(this._right);
  }

  /**
   * Converts to Option, mapping Left to None and Right to Some
   */
  toOption(): Option<R> {
    return new Some(this._right);
  }

  /**
   * Matches on the Either variant
   */
  match<T>(matchers: {
    left: (left: L) => T;
    right: (right: R) => T;
  }): T {
    return matchers.right(this._right);
  }

  /**
   * Executes a function if this is Left
   */
  ifLeft(fn: (left: L) => void): void {
    // Do nothing for Right variant
  }

  /**
   * Executes a function if this is Right
   */
  ifRight(fn: (right: R) => void): void {
    fn(this._right);
  }

  /**
   * Returns a string representation
   */
  toString(): string {
    return `Right(${this._right})`;
  }

  /**
   * Returns JSON representation
   */
  toJSON(): { _tag: "Right"; _right: R } {
    return { _tag: "Right", _right: this._right };
  }
}

/**
 * Creates a Left Either
 */
export function Left<L, R>(left: L): Left<L, R> {
  return new Left(left);
}

/**
 * Creates a Right Either
 */
export function Right<L, R>(right: R): Right<L, R> {
  return new Right(right);
}

/**
 * Wraps a function that might throw in an Either
 */
export function tryCatch<L, R>(
  fn: () => R,
  errorMapper?: (error: unknown) => L
): Either<L, R> {
  try {
    return new Right(fn());
  } catch (error) {
    const mappedError = errorMapper ? errorMapper(error) : (error as L);
    return new Left(mappedError);
  }
}

/**
 * Wraps an async function that might throw in an Either
 */
export async function tryCatchAsync<L, R>(
  fn: () => Promise<R>,
  errorMapper?: (error: unknown) => L
): Promise<Either<L, R>> {
  try {
    const value = await fn();
    return new Right(value);
  } catch (error) {
    const mappedError = errorMapper ? errorMapper(error) : (error as L);
    return new Left(mappedError);
  }
}

/**
 * Combines multiple Eithers into a single Either
 */
export function combine<L, R extends readonly unknown[]>(
  eithers: { [K in keyof R]: Either<L, R[K]> }
): Either<L, R> {
  const values: R = [] as unknown as R;
  
  for (let i = 0; i < eithers.length; i++) {
    const either = eithers[i];
    if (either.isLeft()) {
      return new Left(either._left);
    }
    values[i] = either._right;
  }
  
  return new Right(values);
}

/**
 * Combines multiple Eithers into a single Either, collecting all errors
 */
export function combineAll<L, R extends readonly unknown[]>(
  eithers: { [K in keyof R]: Either<L, R[K]> }
): Either<L[], R> {
  const values: R = [] as unknown as R;
  const errors: L[] = [];
  
  for (let i = 0; i < eithers.length; i++) {
    const either = eithers[i];
    if (either.isLeft()) {
      errors.push(either._left);
    } else {
      values[i] = either._right;
    }
  }
  
  if (errors.length > 0) {
    return new Left(errors);
  }
  
  return new Right(values);
}

/**
 * Filters Eithers, keeping only Right values
 */
export function filterRight<L, R>(eithers: Either<L, R>[]): R[] {
  return eithers
    .filter((either): either is Right<L, R> => either.isRight())
    .map(either => either._right);
}

/**
 * Filters Eithers, keeping only Left values
 */
export function filterLeft<L, R>(eithers: Either<L, R>[]): L[] {
  return eithers
    .filter((either): either is Left<L, R> => either.isLeft())
    .map(either => either._left);
}

/**
 * Partitions Eithers into Right and Left arrays
 */
export function partition<L, R>(eithers: Either<L, R>[]): [R[], L[]] {
  const rightValues: R[] = [];
  const leftValues: L[] = [];
  
  for (const either of eithers) {
    if (either.isRight()) {
      rightValues.push(either._right);
    } else {
      leftValues.push(either._left);
    }
  }
  
  return [rightValues, leftValues];
}

/**
 * Maps over an array of Eithers
 */
export function mapEithers<L, R, S>(
  eithers: Either<L, R>[],
  fn: (value: R) => S
): Either<L, S[]> {
  const mapped: S[] = [];
  
  for (const either of eithers) {
    if (either.isLeft()) {
      return new Left(either._left);
    }
    mapped.push(fn(either._right));
  }
  
  return new Right(mapped);
}

/**
 * Flat maps over an array of Eithers
 */
export function flatMapEithers<L, R, S>(
  eithers: Either<L, R>[],
  fn: (value: R) => Either<L, S>
): Either<L, S[]> {
  const mapped: S[] = [];
  
  for (const either of eithers) {
    if (either.isLeft()) {
      return new Left(either._left);
    }
    const mappedEither = fn(either._right);
    if (mappedEither.isLeft()) {
      return new Left(mappedEither._left);
    }
    mapped.push(mappedEither._right);
  }
  
  return new Right(mapped);
}

/**
 * Reduces an array of Eithers
 */
export function reduceEithers<L, R, S>(
  eithers: Either<L, R>[],
  initialValue: S,
  fn: (accumulator: S, value: R) => S
): Either<L, S> {
  let accumulator = initialValue;
  
  for (const either of eithers) {
    if (either.isLeft()) {
      return new Left(either._left);
    }
    accumulator = fn(accumulator, either._right);
  }
  
  return new Right(accumulator);
}

/**
 * Type guard to check if a value is an Either
 */
export function isEither<L, R>(value: unknown): value is Either<L, R> {
  return (
    typeof value === "object" &&
    value !== null &&
    ("_tag" in value) &&
    (value._tag === "Left" || value._tag === "Right")
  );
}

/**
 * Type guard to check if a value is a Left Either
 */
export function isLeft<L, R>(value: unknown): value is Left<L, R> {
  return isEither(value) && value._tag === "Left";
}

/**
 * Type guard to check if a value is a Right Either
 */
export function isRight<L, R>(value: unknown): value is Right<L, R> {
  return isEither(value) && value._tag === "Right";
}

// Import Result and Option for type compatibility
import type { Result, Ok, Err } from "./result.ts";
import type { Option, Some, None } from "./option.ts";
