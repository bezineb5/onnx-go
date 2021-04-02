package gorgonnx

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type argmaxOp struct {
	dims     int
	along    int // axis
	keepdims bool
}

func newArgmaxOp(dims int, along int, keepdims bool) argmaxOp {
	return argmaxOp{
		dims:     dims,
		along:    along,
		keepdims: keepdims,
	}
}

func (op argmaxOp) Arity() int    { return 1 }
func (op argmaxOp) IsUnary() bool { return true }

func (op argmaxOp) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), tensor.Int)
}

func (op argmaxOp) InferShape(dimsizers ...gorgonia.DimSizer) (tensor.Shape, error) {
	// Exactly 1 input
	if len(dimsizers) != op.Arity() {
		return nil, errors.Errorf("wrong number of arguments for argMaxOp")
	}

	inShape := dimsizers[0].(tensor.Shape)

	if op.along >= inShape.Dims() {
		return nil, fmt.Errorf("shape error, along %d is not a valid axis for shape %v", op.along, inShape)
	}

	if op.keepdims {
		// We keep the dimensions, it's just reduced to 1 item
		shape := inShape.Clone()
		shape[op.along] = 1
		return shape, nil
	}

	// Remove the dimension from the shape
	var dims []int
	for i, d := range inShape {
		if i != op.along {
			dims = append(dims, d)
		}
	}

	// Handle the scalar case
	if len(dims) == 0 {
		return tensor.ScalarShape(), nil
	}

	return tensor.Shape(dims), nil
}

func NewI64(v int64) *gorgonia.I64 { r := gorgonia.I64(v); return &r }

func (op argmaxOp) Do(inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	if len(inputs) != op.Arity() {
		return nil, errors.Errorf("wrong number of arguments for argMaxOp")
	}

	at := inputs[0].(tensor.Tensor)
	switch t := at.(type) {
	case *tensor.Dense:
		var ret *tensor.Dense
		if ret, err = t.Argmax(op.along); err == nil {
			fmt.Printf("Inference output: %#v\n", ret)
			if ret.IsScalar() {
				retVal = NewI64(int64(ret.ScalarValue().(int)))
			} else {
				// Convert tensor from int to int64, as per the ONNX specifications
				if op.keepdims {
					// the tensor reduction ops remove collapsed dimensions, but here we preserve them except in special cases.
					// so we reshape the return to ensure the dimensions match.
					var sh tensor.Shape
					if sh, err = op.InferShape(t.Shape()); err == nil {
						if err = ret.Reshape(sh...); err == nil {
							retVal = ret
						}
					}
				} else {
					retVal = ret
				}
			}
		} else {
			return nil, errors.Wrap(err, "failed to apply *tensor.Dense.Argmax()")
		}
	default:
		return nil, errors.Errorf("Argmax only support tensor.Dense")
	}
	return
}

func (op argmaxOp) ReturnsPtr() bool     { return true }
func (op argmaxOp) OverwritesInput() int { return 0 }
func (op argmaxOp) CallsExtern() bool    { return false }

func (op argmaxOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "argmax-%v", op.along)
}

func (op argmaxOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op argmaxOp) String() string { return fmt.Sprintf("ArgMaxAlong%v", op.along) }

type argmax struct {
	axis     int
	keepdims bool
}

func init() {
	register("ArgMax", newArgMax)
}
func newArgMax() operator {
	return &argmax{}
}

func (a *argmax) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}

	x := children[0].gorgoniaNode

	axis := a.axis
	if axis < 0 {
		axis = x.Dims() - axis
	}

	argmaxOp := newArgmaxOp(x.Dims(), axis, a.keepdims)

	n.gorgoniaNode, err = gorgonia.ApplyOp(argmaxOp, x)
	return err
}

func (a *argmax) init(o onnx.Operation) error {
	a.axis = 0
	a.keepdims = true

	if e, ok := o.Attributes["axis"]; ok {
		if v, ok := e.(int64); ok {
			a.axis = int(v)
		} else {
			return errors.New("axis is not an int64")
		}
	}
	if e, ok := o.Attributes["keepdims"]; ok {
		if v, ok := e.(int64); ok {
			if v == 0 {
				a.keepdims = false
			}
		} else {
			return errors.New("keepdims is not an int64")
		}
	}

	if a.keepdims {
		return errors.New("keepdims must be 0 - any other value is not supported")
	}

	return nil
}
