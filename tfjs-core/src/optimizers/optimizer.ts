/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {dispose, getBackend} from '../globals';
import {variableGrads} from '../gradients';
import {scalar, tensor} from '../ops/ops';
import {Serializable} from '../serialization';
import {Scalar, Variable} from '../tensor';
import {NamedTensor, NamedTensorMap} from '../tensor_types';

/**
 * A variable that belongs to an optimizer.
 *
 * The `originalName` field is required for keeping track of the canonical
 * name of the variable, which is usually the name of the model weight that
 * the variable is related to plus a suffix, e.g., 'dense1/kernel/momentum'.
 * The name of the `Variable` object itself cannot be used directly due to
 * possible deduplication: Every `Variable` must have a unique name but more
 * than one optimizer objects of the same type may be created for the same model
 * or the same `Variable`.
 */
export interface OptimizerVariable {
  originalName: string;
  variable: Variable;
}

/** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
export abstract class Optimizer extends Serializable {
  protected iterations_: number;

  /**
   * Executes `f()` and minimizes the scalar output of `f()` by computing
   * gradients of y with respect to the list of trainable variables provided by
   * `varList`. If no list is provided, it defaults to all trainable variables.
   *
   * @param f The function to execute and whose output to minimize.
   * @param returnCost Whether to return the scalar cost value produced by
   * executing `f()`.
   * @param varList An optional list of variables to update. If specified, only
   * the trainable variables in varList will be updated by minimize. Defaults to
   * all trainable variables.
   *
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  minimize(f: () => Scalar, returnCost = false, varList?: Variable[]): Scalar
      |null {
    const {value, grads} = this.computeGradients(f, varList);

    if (varList != null) {
      const gradArray: NamedTensor[] =
          varList.map(v => ({name: v.name, tensor: grads[v.name]}));
      this.applyGradients(gradArray);
    } else {
      this.applyGradients(grads);
    }

    // Dispose gradients.
    dispose(grads);

    if (returnCost) {
      return value;
    } else {
      value.dispose();
      return null;
    }
  }

  /**
   * Executes `f()` and minimizes the scalar output of `f()` by computing
   * gradients of y with respect to the list of trainable variables provided by
   * `varList`. If no list is provided, it defaults to all trainable variables.
   *
   * @param f The function to execute and whose output to minimize.
   * @param returnCost Whether to return the scalar cost value produced by
   * executing `f()`.
   * @param varList An optional list of variables to update. If specified, only
   * the trainable variables in varList will be updated by minimize. Defaults to
   * all trainable variables.
   *
   * @param peermrContext
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  async minimizeAllReduce(f: () => Scalar,
                          returnCost = false,
                          varList?: Variable[],
                          // tslint:disable-next-line:no-any
                          peermrContext?: any | undefined):
    Promise<Scalar | null> {
    if (!peermrContext) {
      return this.minimize(f, returnCost, varList);
    }

    // compute gradients
    peermrContext.log(`compute gradients start with ${getBackend()}`);

    const {value, grads} = this.computeGradients(f, varList);
    const varNames = Object.keys(grads);
    const nameToN: Map<string, number> = new Map<string, number>();
    const participatingVarNames: string[] = [];
    for (const name of varNames) {
      nameToN.set(name, 1);
      if (grads[name] !== null) {
        participatingVarNames.push(name);
      }
    }
    peermrContext.log(`compute gradients end`);

    // synchronize on iteration count
    peermrContext.log(`vote start ${this.iterations_ || 0}`);
    const globalMaxIterations =
      await peermrContext.vote('iterations', this.iterations_ || 0);
    peermrContext.log(`vote end ${globalMaxIterations}`);

    const workerCount = peermrContext.getWorkerCount();
    const sendRecvOptions = {'direct': true, 'timeout': 30_000};
    const rank: number = peermrContext.getRank();
    // perform allreduce
    let receiveBuffers = [];
    for (let i = 0; i < workerCount - 1; i++) {
      const promises = [];
      for (let j = 0; j < participatingVarNames.length; j++) {
        const name = participatingVarNames[j];
        const tag = [name, i, globalMaxIterations].join('-');
        // send gradients to right neighbor
        if (i === 0) {
          // send this node's gradients to right neighbor
          const gradientsData = await grads[name].data();
          promises.push(peermrContext.sendToRank(
            rank + 1, tag, gradientsData.buffer, sendRecvOptions));
        } else if (receiveBuffers.length > j) {
          // send gradients received from left neighbor to right neighbor
          promises.push(peermrContext.sendToRank(
            rank + 1, tag, receiveBuffers[j], sendRecvOptions));
        } else {
          // did not receive left neighbor's gradients
          // send an averaged gradients
          const gradients = grads[name];
          const N = scalar(nameToN.get(name), grads[name].dtype);
          const sendGradients = gradients.div(N);
          const gradientsData = await sendGradients.data();
          dispose(N);
          dispose(sendGradients);
          promises.push(peermrContext.sendToRank(
            rank + 1, tag, gradientsData.buffer, sendRecvOptions));
        }

        // receive gradients from left neighbor
        promises.push(peermrContext.receiveFromRank(
          rank - 1, tag, sendRecvOptions));
      }

      try {
        peermrContext.log(
          `gradients send/recv start ${i}-${globalMaxIterations}`);
        // tslint:disable-next-line:no-any
        const results: any = await Promise.all(promises);
        peermrContext.log(
          `gradients send/recv end ${i}-${globalMaxIterations}`);
        receiveBuffers = [];
        peermrContext.log(`gradients add start ${i}-${globalMaxIterations}`);
        for (let j = 0; j < participatingVarNames.length; j++) {
          receiveBuffers.push(results[(j * 2) + 1]);
          const name = participatingVarNames[j];
          nameToN.set(name, nameToN.get(name) + 1);
          // convert received ArrayBuffer into appropriate TypedArray
          let receivedTypedArray: Float32Array | Int32Array | Uint8Array;
          const gradients = grads[name];
          switch (gradients.dtype) {
            case 'float32':
            case 'complex64':
              receivedTypedArray = new Float32Array(receiveBuffers[j]);
              break;
            case 'int32':
              receivedTypedArray = new Int32Array(receiveBuffers[j]);
              break;
            case 'bool':
            case 'string':
            default:
              receivedTypedArray = new Uint8Array(receiveBuffers[j]);
          }
          // create tensor gradient from the received TypedArray
          const receivedGradients =
            tensor(receivedTypedArray, gradients.shape, gradients.dtype);
          // add received gradients to this node's gradients
          grads[name] = gradients.add(receivedGradients);
          dispose(gradients);
          dispose(receivedGradients);
        }
        peermrContext.log(`gradients add end ${i}-${globalMaxIterations}`);
      } catch (e) {
        peermrContext.log(`gradients[${i}][${globalMaxIterations}] failed send and receive: ${e}`);
        receiveBuffers = [];
      }
    }

    // average the summed gradients
    peermrContext.log(`gradients average start ${globalMaxIterations}`);
    for (const name of participatingVarNames) {
      const gradients = grads[name];
      const N = scalar(nameToN.get(name), gradients.dtype);
      grads[name] = gradients.div(N);
      dispose(gradients);
      dispose(N);
      peermrContext.log(`gradients ${name} dtype: ${gradients.dtype}`);
    }
    peermrContext.log(`gradients average end ${globalMaxIterations}`);

    peermrContext.log(`gradients apply start ${globalMaxIterations}`);
    if (varList != null) {
      const gradArray: NamedTensor[] =
        varList.map(v => ({name: v.name, tensor: grads[v.name]}));
      this.applyGradients(gradArray);
    } else {
      this.applyGradients(grads);
    }
    peermrContext.log(`gradients apply end ${globalMaxIterations}`);

    dispose(grads);
    if (returnCost) {
      return value;
    } else {
      value.dispose();
      return null;
    }
  }

  /**
   * The number of iterations that this optimizer instance has been invoked for.
   */
  get iterations(): number {
    if (this.iterations_ == null) {
      this.iterations_ = 0;
    }
    return this.iterations_;
  }

  protected incrementIterations() {
    this.iterations_ = this.iterations + 1;
  }

  /**
   * Executes f() and computes the gradient of the scalar output of f() with
   * respect to the list of trainable variables provided by `varList`. If no
   * list is provided, it defaults to all trainable variables.
   *
   * @param f The function to execute and whose output to use for computing
   * gradients with respect to variables.
   * @param varList An optional list of variables to compute gradients with
   * respect to. If specified, only the trainable variables in varList will have
   * gradients computed with respect to. Defaults to all trainable variables.
   *
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  computeGradients(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, grads: NamedTensorMap} {
    return variableGrads(f, varList);
  }

  /**
   * Updates variables by using the computed gradients.
   *
   * @param variableGradients A mapping of variable name to its gradient value.
   *
   * @doc {heading: 'Training', subheading: 'Optimizers'}
   */
  abstract applyGradients(variableGradients: NamedTensorMap|
                          NamedTensor[]): void;

  /**
   * Dispose the variables (if any) owned by this optimizer instance.
   */
  dispose(): void {
    if (this.iterations_ != null) {
      dispose(this.iterations_);
    }
  }

  async saveIterations(): Promise<NamedTensor> {
    if (this.iterations_ == null) {
      this.iterations_ = 0;
    }
    return {
      name: 'iter',  // Named for Python compatibility.
      // TODO(cais): Use 'int64' type when available.
      tensor: scalar(this.iterations_, 'int32')
    };
  }

  async getWeights(): Promise<NamedTensor[]> {
    throw new Error('getWeights() is not implemented for this optimizer yet.');
  }

  async setWeights(weightValues: NamedTensor[]): Promise<void> {
    throw new Error(
        `setWeights() is not implemented for this optimizer class ` +
        `${this.getClassName()}`);
  }

  /**
   * Extract the first element of the weight values and set it
   * as the iterations counter variable of this instance of optimizer.
   *
   * @param weightValues
   * @returns Weight values with the first element consumed and excluded.
   */
  protected async extractIterations(weightValues: NamedTensor[]):
      Promise<NamedTensor[]> {
    this.iterations_ = (await weightValues[0].tensor.data())[0];
    return weightValues.slice(1);
  }
}

Object.defineProperty(Optimizer, Symbol.hasInstance, {
  value: (instance: Optimizer) => {
    return instance.minimize != null && instance.computeGradients != null &&
        instance.applyGradients != null;
  }
});
