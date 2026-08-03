package main

import (
	"C"
)
import (
	"fmt"
	"math"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// Map to store bootstrapping.Evaluators by their slot count
// Initialize the map at package level
var bootstrapperMap = make(map[int]*bootstrapping.Evaluator)

func newBootstrapperParameters(LogPs *C.int, lenLogPs C.int, numSlots C.int) bootstrapping.Parameters {
	logP := CArrayToSlice(LogPs, lenLogPs, convertCIntToInt)
	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(scheme.Params.LogN()),
		LogP:     logP,
		Xs:       scheme.Params.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(numSlots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(*scheme.Params, btpParametersLit)
	if err != nil {
		panic(err)
	}
	return btpParams
}

func installBootstrapper(slots int, btpParams bootstrapping.Parameters, btpKeys *bootstrapping.EvaluationKeys) {
	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	if err != nil {
		panic(err)
	}
	bootstrapperMap[slots] = btpEval
}

//export NewBootstrapper
func NewBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
) {
	slots := int(numSlots)

	if _, exists := bootstrapperMap[slots]; exists {
		return
	}

	btpParams := newBootstrapperParameters(LogPs, lenLogPs, numSlots)

	btpKeys, _, err := btpParams.GenEvaluationKeys(scheme.SecretKey)
	if err != nil {
		panic(err)
	}
	installBootstrapper(slots, btpParams, btpKeys)
}

//export GenerateAndSerializeBootstrapper
func GenerateAndSerializeBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
) (*C.char, C.ulong) {
	btpParams := newBootstrapperParameters(LogPs, lenLogPs, numSlots)
	btpKeys, _, err := btpParams.GenEvaluationKeys(scheme.SecretKey)
	if err != nil {
		panic(err)
	}
	installBootstrapper(int(numSlots), btpParams, btpKeys)

	data, err := btpKeys.MarshalBinary()
	if err != nil {
		panic(err)
	}
	return SliceToCArray(data, convertByteToCChar)
}

//export LoadBootstrapper
func LoadBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
	dataPtr *C.char,
	lenData C.ulong,
) {
	btpParams := newBootstrapperParameters(LogPs, lenLogPs, numSlots)
	data := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))
	btpKeys := new(bootstrapping.EvaluationKeys)
	if err := btpKeys.UnmarshalBinary(data); err != nil {
		panic(err)
	}
	installBootstrapper(int(numSlots), btpParams, btpKeys)
}

//export Bootstrap
func Bootstrap(ciphertextID, numSlots C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	bootstrapper := GetBootstrapper(int(numSlots))

	ctBtp := ctIn.CopyNew()
	ctBtp.LogDimensions.Cols = bootstrapper.LogMaxSlots()

	ctOut, err := bootstrapper.Bootstrap(ctBtp)
	if err != nil {
		panic(err)
	}

	postscale := int(1 << (scheme.Params.LogMaxSlots() - bootstrapper.LogMaxSlots()))
	scheme.Evaluator.Mul(ctOut, postscale, ctOut)

	ctOut.LogDimensions.Cols = scheme.Params.LogMaxSlots()

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

func GetBootstrapper(numSlots int) *bootstrapping.Evaluator {
	bootstrapper, exists := bootstrapperMap[numSlots]
	if !exists {
		panic(fmt.Errorf("no bootstrapper found for slot count: %d", numSlots))
	}
	return bootstrapper
}

//export DeleteBootstrappers
func DeleteBootstrappers() {
	bootstrapperMap = make(map[int]*bootstrapping.Evaluator)
}
