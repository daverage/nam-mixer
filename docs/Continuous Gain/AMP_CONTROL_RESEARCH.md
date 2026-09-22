# How guitar amp gain controls actually work

There is no universal gain-control curve across guitar amplifiers. Most conventional guitar amps use logarithmic (audio-taper) potentiometers for at least some of their gain or volume controls, but this does not mean that the amplifier's actual gain, distortion or output volume follows a logarithmic curve.

For your continuous-gain NAM project, the important distinction is between three different things:

1. The physical taper of the potentiometer.

2. The electrical gain of the amplifier as the potentiometer turns.

3. The resulting sound, including distortion, compression, frequency response and output level.

These are related, but they are not interchangeable.

A continuous-gain model needs to reproduce the third one, rather than merely recreate the potentiometer's electrical taper.


## 1. The three common potentiometer tapers

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(13%2C%2013%2C%2013\)%22%20viewBox%3D%220%200%20340%20235%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20stroke%3D%22currentColor%22%20stroke-width%3D%221%22%20opacity%3D%22.2%22%3E%3Cpath%20d%3D%22M35%2015V195H320%22%2F%3E%3Cpath%20d%3D%22M35%20105H320M35%2015H320M35%20150H320%22%20stroke-dasharray%3D%223%204%22%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%22currentColor%22%20font-size%3D%2210%22%3E%3Ctext%20x%3D%225%22%20y%3D%2219%22%3E100%25%3C%2Ftext%3E%3Ctext%20x%3D%2211%22%20y%3D%22109%22%3E50%25%3C%2Ftext%3E%3Ctext%20x%3D%2217%22%20y%3D%22198%22%3E0%25%3C%2Ftext%3E%3Ctext%20x%3D%2230%22%20y%3D%22211%22%3E0%3C%2Ftext%3E%3Ctext%20x%3D%22171%22%20y%3D%22211%22%3E5%3C%2Ftext%3E%3Ctext%20x%3D%22311%22%20y%3D%22211%22%3E10%3C%2Ftext%3E%3Ctext%20x%3D%22177%22%20y%3D%22229%22%20text-anchor%3D%22middle%22%3EKnob%20position%3C%2Ftext%3E%3Ctext%20x%3D%2236%22%20y%3D%2210%22%3ERelative%20resistance%20%2F%20divider%20fraction%3C%2Ftext%3E%3C%2Fg%3E%3Cpath%20d%3D%22M35%20195L320%2015%22%20fill%3D%22none%22%20stroke%3D%22%232586d9%22%20stroke-width%3D%222.5%22%2F%3E%3Cpath%20d%3D%22M35%20195%20C130%20192%20215%20166%20320%2015%22%20fill%3D%22none%22%20stroke%3D%22%2316a085%22%20stroke-width%3D%222.5%22%2F%3E%3Cpath%20d%3D%22M35%20195%20C140%2048%20235%2018%20320%2015%22%20fill%3D%22none%22%20stroke%3D%22%23db8c21%22%20stroke-width%3D%222.5%22%2F%3E%3C%2Fsvg%3E)

Linear

Audio/log

Reverse audio

Illustrative taper shapes, not measured amplifier gain curves. The relationship can reverse depending on how the pot is wired.

|
Taper

|

Behaviour

|

Typical use

|
| --- | --- | --- |
|

Linear

|

Resistance changes proportionally with rotation.

|

Some gain, tone and bias controls.

|
|

Audio/log

|

Resistance changes slowly initially and then increasingly rapidly.

|

Common for volume and gain controls.

|
|

Reverse audio

|

The opposite distribution of a conventional audio taper.

|

Certain specialised gain and tone circuits.

|

Audio taper and logarithmic taper are generally two names for the same type of control. However, there is no single standard audio-taper curve: different pots can provide substantially different responses at the halfway position.

![](https://www.google.com/s2/favicons?domain=https://guitar.com\&sz=32)

Guitar.com

+1

The circuit surrounding the pot also matters. A linear pot can produce a nonlinear signal response simply through its electrical connections and the surrounding resistors.

![](https://www.google.com/s2/favicons?domain=https://www.analog.com\&sz=32)

Analog Devices

+1

## 2. Why the actual amplifier gain curve is different

Imagine an amplifier with a gain control that changes the signal level entering a distortion stage.

Its behaviour might look like this:

### Illustrative amplifier gain progression

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(13%2C%2013%2C%2013\)%22%20viewBox%3D%220%200%20340%20230%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20stroke%3D%22currentColor%22%20stroke-width%3D%221%22%20opacity%3D%22.2%22%3E%3Cpath%20d%3D%22M35%2015V190H320%22%2F%3E%3Cpath%20d%3D%22M35%20102H320M35%2015H320%22%20stroke-dasharray%3D%223%204%22%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%22currentColor%22%20font-size%3D%2210%22%3E%3Ctext%20x%3D%225%22%20y%3D%2219%22%3EHigh%3C%2Ftext%3E%3Ctext%20x%3D%228%22%20y%3D%22193%22%3ELow%3C%2Ftext%3E%3Ctext%20x%3D%2230%22%20y%3D%22207%22%3E0%3C%2Ftext%3E%3Ctext%20x%3D%22174%22%20y%3D%22207%22%3E5%3C%2Ftext%3E%3Ctext%20x%3D%22310%22%20y%3D%22207%22%3E10%3C%2Ftext%3E%3Ctext%20x%3D%22177%22%20y%3D%22224%22%20text-anchor%3D%22middle%22%3EGain%20knob%20position%3C%2Ftext%3E%3Ctext%20x%3D%2235%22%20y%3D%2210%22%3ERelative%20response%3C%2Ftext%3E%3C%2Fg%3E%3Cpath%20d%3D%22M35%20190%20C75%20185%20107%20160%20145%20120%20S220%2048%20320%2035%22%20fill%3D%22none%22%20stroke%3D%22%232586d9%22%20stroke-width%3D%222.5%22%2F%3E%3Cpath%20d%3D%22M35%20190%20C85%20188%20113%20181%20145%20143%20S195%2035%20238%2028%20S296%2027%20320%2027%22%20fill%3D%22none%22%20stroke%3D%22%23dc8c22%22%20stroke-width%3D%222.5%22%2F%3E%3Cpath%20d%3D%22M35%20190%20C96%20190%20124%20174%20152%20137%20S196%2077%20220%2069%20S275%2067%20320%2067%22%20fill%3D%22none%22%20stroke%3D%22%2316a085%22%20stroke-width%3D%222.5%22%2F%3E%3C%2Fsvg%3E)

Output level

Distortion

Compression

Conceptual curves showing how different aspects of an amp can evolve independently as gain increases.

At low settings, increasing gain might primarily increase output level.

As the following stages start clipping, distortion increases rapidly and output level begins to flatten.

At higher settings, the amplifier might already be heavily saturated. Turning up gain further may change the distortion texture, harmonic content or compression without producing much additional volume.

This is only one possible progression. A clean amplifier, an overdriven vintage amplifier and a modern high-gain amplifier can all behave differently.

There is also the question of where the gain control sits in the circuit. It might attenuate the signal between preamp stages, change the gain of an active stage, or interact with frequency-dependent components. Two amps using identical audio-taper pots can therefore produce completely different gain-control responses.

## 3. What this means for your continuous-gain NAM project

I would separate two things in the model:

## A. Knob-position mapping

How the physical control position translates to the model's conditioning input.

Examples:

* Linear mapping: physical Gain 5 maps to 0.5.

* Audio-taper mapping: physical Gain 5 could map to 0.1, 0.15 or another value, depending on the actual taper.

* Learned mapping: physical Gain 5 maps to a value derived from measured amplifier behaviour.

The correct mapping depends on what the model was trained to interpret as its conditioning input.

## B. Amplifier response

How the model changes its audio behaviour in response to the conditioning input.

The model must learn how the amplifier changes its output level, distortion, compression, frequency response and transient behaviour as the gain control moves.

A log curve applied to the conditioning input cannot, by itself, reproduce these characteristics.

### Should you train using a logarithmic gain scale?

I would not use a universal logarithmic gain scale for all amplifiers.

There are two different situations to consider.

If your gain labels 1 through 10 represent actual physical knob positions, they already contain the effects of the real potentiometer taper. For example, a capture at Gain 5 represents the amp's real response with the knob halfway through its marked range.

Training a model with Gain 5 encoded as 0.5 is therefore reasonable, even if the physical potentiometer is logarithmic. The conditioning value represents the knob position, not the amount of electrical amplification.

On the other hand, if you derive conditioning from measured electrical gain or the fraction of signal passed through a potentiometer, that measurement may require a nonlinear mapping to reproduce the original knob positions.

An audio taper would only be appropriate if you knew it matched the actual circuit and how the conditioning input had been defined.

## 4. A potentially useful experiment: adaptive gain mapping

Your C3, C5 and C10 tests have already demonstrated why different amplifiers might benefit from different capture distributions.

Rather than assuming that three evenly spaced captures are sufficient, I would investigate placing training captures according to how quickly the amplifier's response changes.

Consider two hypothetical amplifiers:

### Amplifier A: Smooth gain progression

The response changes gradually across the full range.

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(13%2C%2013%2C%2013\)%22%20viewBox%3D%220%200%20320%2085%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cpath%20d%3D%22M15%2043H305%22%20stroke%3D%22currentColor%22%20stroke-width%3D%222%22%20opacity%3D%22.3%22%2F%3E%3Cg%3E%3Cpath%20d%3D%22M15%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2215%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E1%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M47.22222222222222%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2247.22222222222222%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E2%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M79.44444444444444%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2279.44444444444444%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E3%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M111.66666666666667%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22111.66666666666667%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E4%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M143.88888888888889%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22143.88888888888889%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E5%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M176.11111111111111%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22176.11111111111111%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E6%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M208.33333333333334%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22208.33333333333334%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E7%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M240.55555555555554%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22240.55555555555554%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E8%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M272.77777777777777%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22272.77777777777777%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E9%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M305%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22305%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E10%3C%2Ftext%3E%3C%2Fg%3E%3Ccircle%20cx%3D%2215%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%232586d9%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%22160%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%232586d9%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%22305%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%232586d9%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ctext%20x%3D%2215%22%20y%3D%2212%22%20fill%3D%22currentColor%22%20font-size%3D%2210%22%3EEvenly%20spaced%20captures%3C%2Ftext%3E%3C%2Fsvg%3E)

Three widely spaced captures might be adequate if the model can accurately learn the response between them.

### Amplifier B: Rapid clean-to-breakup transition

Most of the nonlinear change happens in a narrow region.

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(13%2C%2013%2C%2013\)%22%20viewBox%3D%220%200%20320%2085%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cpath%20d%3D%22M15%2043H305%22%20stroke%3D%22currentColor%22%20stroke-width%3D%222%22%20opacity%3D%22.3%22%2F%3E%3Cg%3E%3Cpath%20d%3D%22M15%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2215%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E1%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M47.22222222222222%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2247.22222222222222%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E2%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M79.44444444444444%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%2279.44444444444444%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E3%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M111.66666666666667%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22111.66666666666667%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E4%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M143.88888888888889%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22143.88888888888889%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E5%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M176.11111111111111%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22176.11111111111111%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E6%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M208.33333333333334%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22208.33333333333334%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E7%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M240.55555555555554%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22240.55555555555554%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E8%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M272.77777777777777%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22272.77777777777777%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E9%3C%2Ftext%3E%3C%2Fg%3E%3Cg%3E%3Cpath%20d%3D%22M305%2039v8%22%20stroke%3D%22currentColor%22%20opacity%3D%22.4%22%2F%3E%3Ctext%20x%3D%22305%22%20y%3D%2265%22%20text-anchor%3D%22middle%22%20font-size%3D%2210%22%20fill%3D%22currentColor%22%3E10%3C%2Ftext%3E%3C%2Fg%3E%3Ccircle%20cx%3D%2215%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%2263.333333333333336%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%2295.55555555555556%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%22127.77777777777777%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%22208.33333333333334%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ccircle%20cx%3D%22305%22%20cy%3D%2243%22%20r%3D%226%22%20fill%3D%22%2316a085%22%20stroke%3D%22white%22%20stroke-width%3D%221%22%2F%3E%3Ctext%20x%3D%2215%22%20y%3D%2212%22%20fill%3D%22currentColor%22%20font-size%3D%2210%22%3ECaptures%20concentrated%20near%20transition%3C%2Ftext%3E%3C%2Fsvg%3E)

Additional captures around the transition might provide more useful training information than captures at higher gain settings where the amplifier response changes relatively little.

This does not necessarily require changing the gain-conditioning architecture. You could initially keep the conditioning input linear with physical knob position and change only where the training captures are taken.

That would isolate the effect of capture placement from the effect of changing the control mapping.

## 5. Could we learn the gain curve automatically?

Yes, and this is the approach I would investigate for your project.

For each amplifier, record a small set of reference measurements across the gain range using the same DI and calibrated input level.

Measure output RMS, spectral characteristics, harmonic distortion, compression and the relationship between input and output levels.

These measurements would help identify where the amplifier changes rapidly and where it remains relatively stable.

You could then use them to select additional training positions and, if appropriate, investigate whether a learned monotonic conditioning mapping improves interpolation.

Crucially, the mapping should not be based solely on output RMS. A gain control can produce significant changes in distortion and compression even when its output level barely changes.

Nor should a learned mapping automatically force every measured characteristic to be monotonic. Real amplifiers can have non-monotonic tonal and dynamic responses.

The resulting model would still have a simple 0–10 gain knob for the user, but its training could be adapted to the behaviour of the particular amplifier.

The key principle is that physical knob position should remain the reference coordinate. The model can learn the nonlinear relationship between that position and the amplifier's sound, rather than imposing an assumed potentiometer taper on every amplifier.
